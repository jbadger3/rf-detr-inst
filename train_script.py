#!/usr/bin/env python3
"""
Training script scaffold for RF-DETR model
"""

import argparse
import os
import sys
from pathlib import Path
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

def get_model_class(model_name):
    """Get the RF-DETR model class based on the model name"""
    if model_name == "nano":
        return RFDETRNano
    elif model_name == "small":
        return RFDETRSmall
    elif model_name == "medium":
        return RFDETRMedium
    elif model_name == "large":
        return RFDETRLarge
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def parse_arguments():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(
        description="Train RF-DETR model on custom dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing training data"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        choices=["nano", "small", "medium", "large"],
        default="nano",
        help="RF-DETR model size to use for training"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loader workers"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save training outputs (checkpoints, logs, etc.)"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate the parsed arguments"""
    # Check if dataset directory exists
    if not os.path.exists(args.dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {args.dataset_dir}")
    
    # Validate numeric arguments
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be positive, got: {args.epochs}")
    
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got: {args.batch_size}")
    
    if args.grad_accum_steps <= 0:
        raise ValueError(f"Gradient accumulation steps must be positive, got: {args.grad_accum_steps}")
    
    if args.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got: {args.lr}")
    
    if args.num_workers < 0:
        raise ValueError(f"Number of workers must be non-negative, got: {args.num_workers}")
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Arguments validated successfully!")



def train_model(args):
    print(f"Setting up training with the following configuration:")
    print(f"  Model: RF-DETR {args.model.upper()}")
    print(f"  Dataset directory: {args.dataset_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Number of workers: {args.num_workers}")
    print(f"  Output directory: {args.output_dir}")

    model_class = get_model_class(args.model)

    model = model_class()
    model.train(dataset_dir=args.dataset_dir, epochs=args.epochs, batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps, lr=args.lr, num_workers=args.num_workers,
                output_dir=args.output_dir)


def main():
    """Main function"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Validate arguments
        validate_arguments(args)
        
        # Start training
        train_model(args)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
