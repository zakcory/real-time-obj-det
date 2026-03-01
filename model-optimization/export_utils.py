import torch
import sys
from enum import Enum

class ModelType(Enum):
    YOLOV9 = "YOLOV9"
    DINOV3 = "DINOv3"

def get_yolov9_model(model_source_path: str, model_path: str) -> tuple[torch.nn.Module, torch.Tensor]:
    class YOLOV9Wrapper(torch.nn.Module):
        """ Wrapper for YOLOv9 model to adjust output format """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            output = self.model(x)
            # Extract BBOXes only
            return output[0]

    # Add model dependencies to path
    sys.path.insert(0, model_source_path)

    # Load model
    model_base = torch.load(model_path, weights_only=False)
    model = YOLOV9Wrapper(model_base['model']).eval()

    return model

def get_dinov3_model(model_source_code: str, model_path: str, dino_type: str) -> tuple[torch.nn.Module, torch.Tensor]:
    class DINOV3Wrapper(torch.nn.Module):
        """Wrapper to extract CLS token from DINOv3 model output"""
        
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            # Get the full output from DINOv3
            output = self.model.forward_features(x)
            
            # Extract CLS token (first token in sequence)
            cls_token = output['x_norm_clstoken']
            
            return cls_token

    # Load model using torch.hub
    model_base = torch.hub.load(
        model_source_code,
        dino_type,
        source='local',
        pretrained=False
    )

    # Load weights seperately
    state_dict = torch.load(
        model_path, 
        map_location='cpu', 
        weights_only=True
    )
    model_base.load_state_dict(state_dict)
    
    # Wrap model to extract CLS token and set to eval mode
    model = DINOV3Wrapper(model_base).eval()

    return model