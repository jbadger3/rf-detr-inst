from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
import torch
import coremltools as ct
from PIL import Image
import torchvision.transforms.v2 as transforms
import numpy as np
import argparse
import os

def preprocess_image(pil_image, target_size=384):
    """
    Preprocess a PIL image for model inference.
    
    Args:
        pil_image (PIL.Image): Input PIL image
        target_size (int): Target size for resizing (default: 384)
        
    Returns:
        torch.Tensor: Preprocessed tensor with shape (1, 3, target_size, target_size)
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToImage(),  # Converts PIL to tensor
        transforms.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(pil_image).unsqueeze(0)  # Shape: (1, 3, target_size, target_size)
    return tensor

def get_model_class(model_size):
    """
    Get the appropriate RFDETR model class based on size.
    
    Args:
        model_size (str): Model size ('nano', 'small', 'medium', 'base', 'large')
        
    Returns:
        RFDETR class: The corresponding model class
    """
    model_map = {
        'nano': RFDETRNano,
        'small': RFDETRSmall,
        'medium': RFDETRMedium,
        'base': RFDETRBase,
        'large': RFDETRLarge
    }
    return model_map[model_size]

def get_model_resolution(model_size):
    """
    Get the resolution for each model size based on config.
    
    Args:
        model_size (str): Model size ('nano', 'small', 'medium', 'base', 'large')
        
    Returns:
        int: The resolution for the model
    """
    resolution_map = {
        'nano': 384,
        'small': 512,
        'medium': 576,
        'base': 560,
        'large': 560
    }
    return resolution_map[model_size]

def parse_arguments():
    """
    Parse command line arguments for CoreML export.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Export RF-DETR model to CoreML format')
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['nano', 'small', 'medium', 'base', 'large'],
        default='nano',
        help='RF-DETR model size (default: nano)'
    )
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='output/checkpoint_best_regular.pth',
        help='Path to checkpoint .pth file (default: output/checkpoint_best_regular.pth)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='output',
        help='Output directory to save the CoreML model (default: output)'
    )
    
    parser.add_argument(
        '--version', 
        type=str, 
        default='1.0.0',
        help='Version string for the model (default: 1.0.0)'
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create output directory and subdirectories if needed
    os.makedirs(args.output, parents=True, exist_ok=True)
    
    # Get the appropriate model class and resolution
    ModelClass = get_model_class(args.model)
    resolution = get_model_resolution(args.model)
    
    # Initialize model with checkpoint
    model = ModelClass(pretrain_weights=args.checkpoint, num_classes=5)
    sample_inputs = torch.rand(1, 3, resolution, resolution).cuda()
    model.optimize_for_inference(compile=True)

    #CoreML uses scale and bias for preprocessing params
    #see https://apple.github.io/coremltools/docs-guides/source/image-inputs.html
    scale = 1/(0.226*255.0)
    bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

    model_from_trace = ct.convert(
        model.model.inference_model,
        inputs=[ct.ImageType(
                    name="input",
                    shape=sample_inputs.shape, 
                    scale=scale, bias=bias,
                    )
                ],
        outputs=[ct.TensorType(dtype=np.float16), ct.TensorType(dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16  
    )
    
    # Set model metadata
    model_from_trace.short_description = f"RF-DETR {args.model.title()} object detection model"
    model_from_trace.version = args.version
    
    # Save the model
    output_path = os.path.join(args.output, f"rfdetr_{args.model}_v{args.version}.mlpackage")
    model_from_trace.save(output_path)
    
    print(f"CoreML model saved to: {output_path}")
    print(f"Model: RF-DETR {args.model.title()}")
    print(f"Version: {args.version}")
    print(f"Checkpoint: {args.checkpoint}")

if __name__ == "__main__":
    main()