import torch
from rfdetr import RFDETRNano
from rfdetr import config
from torch.nn.attention import SDPBackend, sdpa_kernel
checkpoint_file = 'output/checkpoint_best_regular.pth'
num_classes = 5
import ai_edge_torch
import tensorflow as tf
sample_inputs = (torch.rand(1, 3, 384, 384).cuda(),)
model = RFDETRNano(pretrain_weights=checkpoint_file, num_classes=num_classes)
config.init([[24, 24]])
#torch._logging.set_logs(dynamic=10)
model.optimize_for_inference(compile=False)
model_mod = model.model.model
sample_inputs = (torch.rand(1, 3, 384, 384).cuda(),)
tfl_converter_flags =  {
    'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]},
    'optimizations': [tf.lite.Optimize.DEFAULT]
}
with sdpa_kernel(SDPBackend.MATH):
    edge_model = ai_edge_torch.convert(model.model.inference_model.eval(), sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
    edge_model.export("rfdetr.tflite")
