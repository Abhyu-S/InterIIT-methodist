# Key fixes for your original ViT code (though success is not guaranteed)

class LitViTQAT(pl.LightningModule):
    def __init__(self, num_classes: int = 100, lr: float = 5e-6, weight_decay: float = 0.01, use_qat: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self._qat_prepared = False
        self.use_qat = use_qat
        # Store original state dict for later loading
        self._original_state_dict = None

    def _prepare_qat_eager_fixed(self):
        """Fixed eager mode preparation."""
        try:
            # Properly wrap the model
            class QuantWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.quant = torch.ao.quantization.QuantStub()
                    self.model = model
                    self.dequant = torch.ao.quantization.DeQuantStub()
                    
                def forward(self, x):
                    x = self.quant(x)
                    x = self.model(x)
                    x = self.dequant(x)
                    return x
            
            # Save state dict before wrapping
            state_dict = self.model.state_dict()
            
            # Wrap model
            wrapped_model = QuantWrapper(self.model)
            
            # Set qconfig
            wrapped_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            
            # Prepare QAT
            torch.ao.quantization.prepare_qat(wrapped_model, inplace=True)
            
            # Try to load state dict into the inner model
            # This is tricky and may fail
            try:
                wrapped_model.model.load_state_dict(state_dict, strict=False)
            except:
                pass  # State dict mismatch is expected
            
            print("Eager-mode QAT preparation completed (experimental)")
            return wrapped_model
            
        except Exception as e:
            print(f"Eager QAT preparation failed: {e}")
            return None

    def on_fit_start(self) -> None:
        if not self.use_qat or self._qat_prepared:
            return

        # Save original state dict
        self._original_state_dict = self.model.state_dict()
        
        print("WARNING: ViT models are not well-suited for quantization!")
        print("Attempting experimental QAT preparation...")
        
        self.model.train()
        self.model = self.model.cpu()  # QAT prep must be on CPU
        
        # Only try eager mode for ViT (FX will definitely fail)
        prepared = self._prepare_qat_eager_fixed()
        
        if prepared is None:
            print("ERROR: Could not prepare ViT for QAT. Falling back to FP32 training.")
            self.use_qat = False
            self.model = self.model.to(self.device)
            return
        
        self.model = prepared.to(self.device)
        self._qat_prepared = True
        print("Model prepared for QAT (experimental - may not work correctly)")

# Additional fix for conversion
def convert_to_int8_safe(prepared_model):
    """Safer INT8 conversion with fallbacks."""
    try:
        # Try standard conversion
        prepared_model.eval()
        prepared_model = prepared_model.cpu()
        model_int8 = torch.ao.quantization.convert(prepared_model, inplace=False)
        print("INT8 conversion succeeded")
        return model_int8
    except Exception as e:
        print(f"INT8 conversion failed: {e}")
        print("Returning fake-quantized model (not true INT8)")
        return prepared_model