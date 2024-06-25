# ONNX Model Conversion and Inference

This project provides scripts to convert TensorFlow, Keras, and PyTorch models to ONNX format and perform inference using ONNX Runtime.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/CODIN14/ONNX-Conversion-Toolkit-and-inferencing.git
cd ONNX-Conversion-Toolkit-and-inferencing
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```

## Usage
### Convert TensorFlow Model to ONNX
from convert import convert_tensorflow_to_onnx

tf_model_path = "path_to_tf_model"
onnx_model_path = "path_to_save_onnx_model"
convert_tensorflow_to_onnx(tf_model_path, onnx_model_path)

### Convert Keras Model to ONNX
from convert import convert_keras_to_onnx

keras_model_path = "path_to_keras_model"
onnx_model_path = "path_to_save_onnx_model"
convert_keras_to_onnx(keras_model_path, onnx_model_path)

### Convert PyTorch Model to ONNX
from convert import convert_pytorch_to_onnx

pytorch_model_path = "path_to_pytorch_model.pth"
onnx_model_path = "path_to_save_onnx_model.onnx"
convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path)

### Run Inference Using ONNX Runtime
import numpy as np
from inference import run_inference_onnx

onnx_model_path = "path_to_onnx_model.onnx"
input_image = np.random.randn(1, 1, 28, 28).astype(np.float32)

output = run_inference_onnx(onnx_model_path, input_image)
print("Inference Output:", output)

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.



