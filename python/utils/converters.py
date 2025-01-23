import torch
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

import tensorflow as tf 
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import  Input, BatchNormalization, Dense, Concatenate, Conv2D
import numpy as np


def keras2tflite(model, input_shape=None, num_threads=1):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=num_threads)
    if input_shape is not None:
        interpreter.resize_tensor_input(0, input_shape, strict=True)
    return interpreter

def pytorch2tflite(model, tmp_folder='./tmp/', input_shape=(1,1,96,96), quantize=False, num_threads=1):
    sample_input = torch.rand(input_shape)
    torch.onnx.export(
        model,                  # PyTorch Model
        sample_input,                    # Input tensor
        './tmp/tmp.onnx',        # Output file (eg. 'output_model.onnx')
        opset_version=12,       # Operator support version
        input_names=['input'],   # Input tensor name (arbitary)
        output_names=['output'], # Output tensor name (arbitary)
        dynamic_axes={'input' : {2 : 'H', 3: 'W'},    # variable length axes
                    'output' : {2 : 'H', 3: 'W'}})

    onnx_model = onnx.load(f"{tmp_folder}tmp.onnx")
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(f'{tmp_folder}tmp.pb')
    converter = tf.lite.TFLiteConverter.from_saved_model(f'{tmp_folder}tmp.pb')
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable LiteRT ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open(f'{tmp_folder}tmp.tflite', 'wb') as f:
        f.write(tflite_model)

    return tf.lite.Interpreter(
        model_path=f"{tmp_folder}tmp.tflite", num_threads=num_threads
    )


