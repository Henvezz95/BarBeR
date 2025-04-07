import torch
from torch import nn
import numpy as np
import onnxruntime
from typing import Any
import os
from utils.activations import act_dict

def get_first_conv_layer(model):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return module
    return None

class ModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_threads: int = 1, device: str = 'cpu', activation: str = 'sigmoid'):
        super().__init__()
        self.model = model.eval()
        self.onnx_session = None
        self.num_threads = num_threads
        self.device = device
        self.activation = act_dict[activation]
        in_ch = get_first_conv_layer(model).in_channels
        self.input_tensor = torch.randn(1, in_ch, 224,224)
        if device == 'cpu':
            self.model.to('cpu')
            self.activation.to('cpu')
        elif device in {'gpu', 'cuda'}:
            self.model.to('cuda')
            self.activation.to('cuda')

    def convert2onnx(self, onnx_filename = "./Saved Models/tmp.onnx"):
        torch.onnx.export(
            self.model, 
            self.input_tensor, 
            onnx_filename, 
            opset_version=17, 
            do_constant_folding=True, 
            input_names=["input"], 
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},  # Dynamic batch & spatial size
                "output": {0: "batch_size", 2: "height", 3: "width"}  # Dynamic batch & spatial size
            }
        )
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = self.num_threads
        if self.device == 'cpu':
            self.onnx_session = onnxruntime.InferenceSession(onnx_filename, 
                                                             sess_options, 
                                                             providers=["CPUExecutionProvider"])
        else:
            self.onnx_session = onnxruntime.InferenceSession(onnx_filename, 
                                                             sess_options, 
                                                             providers=["GPUExecutionProvider"])
        os.remove(onnx_filename)
            

    def convert2pytorch(self):
        self.onnx_session = None

    def forward(self, x):
        if self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: x})[0]
            outputs = self.activation(torch.from_numpy(outputs))
        else:
            x_torch = torch.from_numpy(x)
            if self.device in ['gpu', 'cuda']:
                outputs = self.model(x_torch.cuda())
            else:
                outputs = self.model(x_torch)
            outputs = self.activation(outputs)
        return outputs.detach().cpu().numpy()