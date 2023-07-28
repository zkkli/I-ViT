import argparse
import torch
import tvm
from tvm import relay
from tvm import target
from tvm.contrib.download import download_testdata

from torchvision import transforms
from PIL import Image
import numpy as np

import models.build_model as build_model
from models.layers import QuantizeContext

import convert_model


parser = argparse.ArgumentParser(description="TVM-Accuracy")

parser.add_argument("--model-name", default='deit_tiny_patch16_224',
                    choices=['deit_tiny_patch16_224',
                             'deit_small_patch16_224',
                             'deit_base_patch16_224'],
                    help="model fullname")
parser.add_argument("--model-path", default='',
                    help="saved checkpoint path in QAT (checkpoint.pth.tar)")
parser.add_argument("--params-path", default='',
                    help="saved parameters path in convert_model.py (params.npy)")


def main():
    args = parser.parse_args()

    # Set target device
    target = 'cuda'

    # Load params
    model = torch.load(args.model_path)
    pretrained_params = np.load(args.params_path, allow_pickle=True)[()]
    depth = 12
    convert_model.load_qconfig(model, depth)

    # Classic cat example!
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))
    # Preprocess the image and convert to tensor
    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    input_image = np.expand_dims(img, 0)
    input_image = input_image / QuantizeContext.qconfig_dict['qconfig_embed_conv'].input_scale
    input_image = np.clip(input_image, -128, 127)
    input_image = np.round(input_image)
    input_image = input_image.astype("int8")

    # Load model
    name = args.model_name
    batch_size = 1
    shape = list(input_image.shape)
    image_shape = (3, 224, 224)
    data_layout = "NCHW"
    kernel_layout = "OIHW"
    func, params = build_model.get_workload(name=name,
                                            batch_size=batch_size,
                                            image_shape=image_shape,
                                            dtype="int8",
                                            data_layout=data_layout,
                                            kernel_layout=kernel_layout)

    # Build model
    pretrained_params = {**pretrained_params}
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=pretrained_params)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(target, 0)))

    # Run model
    input_data = np.repeat(input_image, batch_size, axis=0)
    runtime.set_input('data', input_data)
    runtime.run()

    tvm_result = runtime.get_output(0).numpy()

    tvm_top1_labels = np.argsort(tvm_result[0])[::-1][:5]
    print("TVM top1 labels:", tvm_top1_labels)


if __name__ == "__main__":
    main()