#!/usr/bin/env python3
"""
train_lightning_qat.py

Upgraded: Now also reports all model parameters (names, shapes, dtypes, requires_grad)
for FP32, QAT-prepared, and INT8 models.

Converted from your notebook(s) into a PyTorch Lightning-ready script.
"""

import argparse
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization.quantize_fx as quantize_fx
import timm
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

# --------------------------- Utilities ---------------------------------

def get_model_size_mb(model: torch.nn.Module) -> float:
    param_size = 0
    buffer_size = 0
    for p in model.parameters():
        param_size += p.nelement() * p.element_size()
    for b in model.buffers():
        buffer_size += b.nelement() * b.element_size()
    return float((param_size + buffer_size) / 1024 ** 2)


def dump_model_parameters(model: torch.nn.Module, out_path: str, stage: str):
    """Dump model parameters with names, shapes, dtypes, requires_grad."""
    lines = [f"Model parameters dump ({stage}):\n"]
    for name, param in model.named_parameters():
        lines.append(
            f"{name}: shape={tuple(param.shape)}, dtype={param.dtype}, requires_grad={param.requires_grad}, numel={param.numel()}"
        )
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Saved {stage} parameter dump to {out_path}")


# --------------------------- Data --------------------------------------

def prepare_data_loaders(batch_size: int = 16, num_workers: int = 4, data_dir: str = "./data"):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# --------------------------- Lightning Module ---------------------------
class LitViTQAT(pl.LightningModule):
    def __init__(self, num_classes: int = 100, lr: float = 5e-6, weight_decay: float = 0.01, use_qat: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self._qat_prepared = False
        self._use_eager_qat = False  # if FX fails we'll fall back to eager-mode QAT
        self.use_qat = use_qat

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/acc', acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return opt

    def _prepare_qat_fx_with_tracing(self, example_input: torch.Tensor, qconfig_dict: dict):
        """Try prepare_qat_fx directly; if it fails, attempt to fx.trace (symbolic_trace) first."""
        try:
            prepared = quantize_fx.prepare_qat_fx(self.model, qconfig_dict, example_inputs=example_input)
            print("prepare_qat_fx succeeded on original model.")
            return prepared
        except Exception as e:
            print("prepare_qat_fx failed on original model:", e)
            # Try symbolic_trace then prepare
            try:
                import torch.fx as fx
                print("Attempting torch.fx.symbolic_trace on model...")
                traced = fx.symbolic_trace(self.model)
                prepared = quantize_fx.prepare_qat_fx(traced, qconfig_dict, example_inputs=example_input)
                print("prepare_qat_fx succeeded on traced model.")
                return prepared
            except Exception as e2:
                print("symbolic_trace or prepare_qat_fx on traced model failed:", e2)
                return None

    def _prepare_qat_eager(self, example_input: torch.Tensor, qconfig_dict: dict):
        """Fallback: convert to eager-mode QAT by inserting QuantStub/DeQuantStub and using prepare_qat.
        This is more manual but works with modules that aren't FX-traceable."""
        try:
            # Ensure QuantStub/DeQuantStub exist
            class EagerWrapper(nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.quant = torch.ao.quantization.QuantStub()
                    self.inner = inner
                    self.dequant = torch.ao.quantization.DeQuantStub()

                def forward(self, x):
                    x = self.quant(x)
                    x = self.inner(x)
                    x = self.dequant(x)
                    return x

            wrapped = EagerWrapper(self.model)
            # Prepare eager-mode QAT (module-based)
            torch.ao.quantization.prepare_qat(wrapped, inplace=True)
            self._use_eager_qat = True
            print("Eager-mode prepare_qat succeeded.")
            return wrapped
        except Exception as e:
            print("Eager-mode prepare_qat failed:", e)
            return None

    def on_fit_start(self) -> None:
        if not self.use_qat or self._qat_prepared:
            return

        example_input = torch.randn(1, 3, 224, 224).to(self.device)
        qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        qconfig_dict = {"": qconfig}

        print("Preparing model for QAT (trying FX first). This may take a moment...")
        self.model.train()

        # Try FX-based flow (preferred)
        prepared = self._prepare_qat_fx_with_tracing(example_input, qconfig_dict)

        if prepared is None:
            # Try eager-mode fallback
            print("Falling back to eager-mode QAT (module-based) because FX couldn't prepare the model.")
            prepared = self._prepare_qat_eager(example_input, qconfig_dict)

        if prepared is None:
            raise RuntimeError("Could not prepare model for QAT with either FX or eager-mode workflows.")

        # Replace model with prepared version (could be GraphModule or nn.Module wrapper)
        self.model = prepared.to(self.device)
        self._qat_prepared = True
        print("Model prepared for QAT.Model type:", type(self.model))


# --------------------------- Evaluation Helpers ------------------------

def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int = 100):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    inf_times = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            start = time.time()
            outputs = model(data)
            end = time.time()
            inf_times.append(end - start)
            _, preds = torch.max(outputs, 1)
            total += target.size(0)
            correct += (preds == target).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = correct / total
    avg_inf = float(np.mean(inf_times))
    return acc, avg_inf, all_preds, all_targets


# --------------------------- Main flow --------------------------------

def main_cli(args: argparse.Namespace):
    pl.seed_everything(42)

    train_loader, test_loader = prepare_data_loaders(batch_size=args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    logger = CSVLogger(save_dir=args.log_dir, name='vit_qat')
    ckpt_cb = ModelCheckpoint(monitor='val/acc', mode='max', save_top_k=1, filename='vit-qat-{epoch:02d}-{val/acc:.4f}')
    lrmon = LearningRateMonitor(logging_interval='step')

    # 1) Train FP32
    model_fp32_module = LitViTQAT(num_classes=args.num_classes, lr=args.lr, weight_decay=args.weight_decay, use_qat=False)
    trainer_fp32 = pl.Trainer(max_epochs=args.fp32_epochs, accelerator='gpu' if torch.cuda.is_available() else None, devices=args.devices if torch.cuda.is_available() else None, logger=logger, callbacks=[ckpt_cb, lrmon], default_root_dir=args.work_dir)
    trainer_fp32.fit(model_fp32_module, train_loader, test_loader)

    fp32_ckpt_path = os.path.join(args.work_dir, 'vit_large_fp32_cifar100.pth')
    torch.save(model_fp32_module.model.state_dict(), fp32_ckpt_path)
    print(f"Saved FP32 model state_dict to {fp32_ckpt_path}")

    dump_model_parameters(model_fp32_module.model, os.path.join(args.work_dir, "params_fp32.txt"), stage="FP32")

    model_fp32_module.model.to(device)
    model_fp32_module.model.eval()
    fp32_acc, fp32_inf, _, _ = evaluate(model_fp32_module.model, test_loader, device, args.num_classes)
    print(f"FP32 eval accuracy: {fp32_acc:.4f}, avg inf time: {fp32_inf:.4f}s, size: {get_model_size_mb(model_fp32_module.model):.2f} MB")

    # 2) QAT fine-tune
    model_qat_module = LitViTQAT(num_classes=args.num_classes, lr=args.lr_qat, weight_decay=args.weight_decay, use_qat=True)
    model_qat_module.model.load_state_dict(torch.load(fp32_ckpt_path, map_location='cpu'))

    trainer_qat = pl.Trainer(max_epochs=args.qat_epochs, accelerator='gpu' if torch.cuda.is_available() else None, devices=args.devices if torch.cuda.is_available() else None, logger=logger, callbacks=[ckpt_cb, lrmon], default_root_dir=args.work_dir, accumulate_grad_batches=args.accumulate_grad_batches)
    trainer_qat.fit(model_qat_module, train_loader, test_loader)

    dump_model_parameters(model_qat_module.model, os.path.join(args.work_dir, "params_qat_prepared.txt"), stage="QAT-prepared")

    prepared_model = model_qat_module.model.to('cpu')
    prepared_model.eval()
    print("Converting prepared QAT model to INT8 (convert_fx or eager convert)...")

    model_int8 = None
    # If the prepared model is an FX GraphModule -> use convert_fx
    import torch.fx as fx
    if isinstance(prepared_model, fx.GraphModule):
        try:
            model_int8 = quantize_fx.convert_fx(prepared_model)
            print("convert_fx succeeded on GraphModule.")
        except Exception as e:
            print("convert_fx on GraphModule failed:", e)
            model_int8 = None
    else:
        # If we used eager-mode QAT wrapper, try eager convert
        if getattr(model_qat_module, '_use_eager_qat', False):
            try:
                model_int8 = torch.ao.quantization.convert(prepared_model.eval(), inplace=False)
                print("Eager-mode convert succeeded.")
            except Exception as e:
                print("Eager-mode convert failed:", e)
                model_int8 = None
        else:
            # Try to symbolically trace the prepared model and then run convert_fx
            try:
                print("Prepared model is not a GraphModule; attempting symbolic_trace(prepared_model) and convert_fx...")
                traced = fx.symbolic_trace(prepared_model)
                model_int8 = quantize_fx.convert_fx(traced)
                print("convert_fx succeeded on traced prepared model.")
            except Exception as e:
                print("symbolic_trace/convert_fx failed:", e)
                model_int8 = None

    if model_int8 is None:
        print("INT8 conversion failed with all strategies. Proceeding with the prepared (fake-quant) model for evaluation/saving.")
        model_int8 = prepared_model


    dump_model_parameters(model_int8, os.path.join(args.work_dir, "params_int8.txt"), stage="INT8")

    int8_acc, int8_inf, _, _ = evaluate(model_int8, test_loader, torch.device('cpu'), args.num_classes)
    print(f"INT8 eval accuracy: {int8_acc:.4f}, avg inf time (cpu): {int8_inf:.4f}s, size: {get_model_size_mb(model_int8):.2f} MB")

    int8_path = os.path.join(args.work_dir, 'vit_large_qat_int8_cifar100.pth')
    torch.save(model_int8.state_dict(), int8_path)
    print(f"Saved INT8 model state_dict to {int8_path}")

    print("Done.")


# --------------------------- CLI --------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Lightning QAT ViT Large for CIFAR-100 with parameter dumps')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--work_dir', type=str, default='./outputs')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--fp32_epochs', type=int, default=2)
    parser.add_argument('--qat_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--lr_qat', type=float, default=5e-7)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None,
                    help="Path to Lightning checkpoint to resume training")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main_cli(args)
