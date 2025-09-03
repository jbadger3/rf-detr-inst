# LiteRT (TFLite) Model Conversion Notes

## Overview
Converting PyTorch models to LiteRT used to follow the pathway: PyTorch -> Onnx -> TFLite.  The old way of doing things often led to suboptimizd models and for this project has not been met with much success.  See [this issue with a closed PR](https://github.com/roboflow/rf-detr/issues/173) and [this post](https://github.com/roboflow/rf-detr/issues/289) which also discusses challenges associated with model quantization.

This branch uses [ai_edge_torch](https://github.com/google-ai-edge/ai-edge-torch), a package developed by Google for model conversion. Under the hood the new pathways is PyTorch --> export using Torch Dynamo to an exported program --> StableHLO (high level operations) encoded using MLIR (a binary multi-level intermediate representation) --> TFLite flatbuffer.  

The intermediate/bridging layer comes from [OpenXLA](https://openxla.org/) (XLA = accelerated linear algebra), which I believe was originally designed to help transport code for use on TPUs from the various machine learnign computational libraries, but now also includes a suite of tools to make it easier to port models from research to device. [StableHLO](https://openxla.org/stablehlo) is basically an intermediate layer between the original model that allows for operations versioning, backwards compatiblity, etc.

## Conversion notes

### Dynamically shaped inputs

Dynamically shaped inputs (e.g. from tensors) are a big pain in the but for the conversion process.  This makes sense because final model needs to be turned into a single graph with prespecified inputs and outputs for each node.  Dynamic shapes and branching from if - else statements add complexity and ambiguity.  In particular, for this model, the spatial shapes varied depending on which size of model was being exported.  Torch export and JAX handle dynamic shapes a bit differently though. I ran into problems after adding various torch._check statments.  Torch export would succeed, but JAX would complain about those 'fixes' downstream.  In the end the spatial shapes were added to config.py to solve the ambiguity.

### Unsupported ops

Scaled dot product attention had to be manually implemented and various slicing and other operations had to be reformulated from random errors.

### Current status

The export and lowering process completes, but the torch.topk operation connot be translated to a suitable rtlite op. See [this feature request](https://github.com/google-ai-edge/ai-edge-torch/issues/555).