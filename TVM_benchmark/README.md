## I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference

Below are instructions for performing integer-only inference of DeiT models on 2080Ti GPU, with separate accuracy and speed evaluations.

## 0. Install TVM
- You can follow the official [tutorial](https://tvm.apache.org/docs/install/from_source.html#install-from-source) to install TVM.

## 1. Convert Model
- Save checkpoint of QAT in PyTorch (checkpoint.pth.tar)
- Convert Pytorch parameters to TVM parameters (params.npy):
```bash
python convert_model --model-path <path-to-checkpoint> --params-path <path-to-save-params>

Required arguments:
 <path-to-checkpoint> : Path to saved checkpoint of QAT (checkpoint.pth.tar)
 <path-to-save-params> : Path to save TVM parameters
```

## 2. Evaluation
### 2.1 Accuracy
- You can evaluate the accuracy of a model using the following command:

```bash
python evaluate_accuracy.py --model-name <model-name> --model-path <path-to-checkpoint> --params-path <path-to-params>

Required arguments:
 <model-name> : Model name, e.g., 'deit_small_patch16_224'
 <path-to-checkpoint> : Path to saved checkpoint of QAT (checkpoint.pth.tar)
 <path-to-params> : Path to saved TVM parameters (params.npy)
```

### 2.2 Latency
- You can perform **TVM auto-tuning** and evaluate the latency of a model using the following command:

```bash
python evaluate_latency.py --model-name <model-name>  --log-path <path-to-save-log> --target <tvm-target>

Required arguments:
 <model-name> : Model name, e.g., 'deit_small_patch16_224'
 <path-to-save-log> : Path to save tuning log
 <tvm-target> : TVM target, e.g., 'cuda -model=2080ti'
```

## Citation

We appreciate it if you would please cite the following paper if you found the implementation useful for your work:

```bash
@article{li2022ivit,
  title={I-ViT: integer-only quantization for efficient vision transformer inference},
  author={Li, Zhikai and Gu, Qingyi},
  journal={arXiv preprint arXiv:2207.01405},
  year={2022}
}
```
