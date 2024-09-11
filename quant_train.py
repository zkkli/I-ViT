import argparse
import os
import time
import math
import logging
import numpy as np

import torch
import torch.nn as nn

from timm.utils import accuracy
import timm
from models import *
from utils import *


parser = argparse.ArgumentParser(description="I-ViT")

parser.add_argument(
    "--model",
    default="deit_tiny",
    choices=[
        "deit_tiny",  # soft 15, gelu 23
        "deit_small",  # soft 15, gelu 29
        "deit_base",
        "vit_base",
        "vit_large",
        "swin_tiny",
        "swin_small",
        "swin_base",
    ],
    help="model",
)
parser.add_argument(
    "--dataset", metavar="DIR", default="data/ImageNet", help="path to dataset"
)
parser.add_argument("--nb-classes", default=1000, type=int, help="number of classes")
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument(
    "--output-dir",
    type=str,
    default="results/",
    help="path to save log and quantized model",
)
parser.add_argument("--calib_batchsize", default=32, type=int)
parser.add_argument("--calib_images", default=1024, type=int)
parser.add_argument("--val_batchsize", default=128, type=int)
parser.add_argument("--num_workers", default=8, type=int)

parser.add_argument("--intsoftmax_exp_n", default=15, type=int)
parser.add_argument("--intgelu_exp_n", default=23, type=int)


parser.add_argument(
    "--attn_quant",
    default="Log2_Int_Quantizer",
    choices=[
        "Symmetric_UINT4",
        "Symmetric_UINT8",
        "Log2_half_Int_Quantizer",
        "Log2_Int_Quantizer",
        "Log2Quantizer",
        "LogSqrt2Quantizer",
        "NoQuant",
    ],
    help="attention quantization. only 4bit quantization is supported",
)


def str2model(name):
    d = {
        "deit_tiny": deit_tiny_patch16_224,
        "deit_small": deit_small_patch16_224,
        "deit_base": deit_base_patch16_224,
        "vit_base": vit_base_patch16_224,
        "vit_large": vit_large_patch16_224,
        "swin_tiny": swin_tiny_patch4_window7_224,
        "swin_small": swin_small_patch4_window7_224,
        "swin_base": swin_base_patch4_window7_224,
    }
    print("Model: %s" % d[name].__name__)
    return d[name]


def main():
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    import warnings

    warnings.filterwarnings("ignore")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        filename=args.output_dir + "log.log",
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    device = torch.device(args.device)

    # Dataset
    train_loader, val_loader = build_dataset(args)

    # Model
    model = str2model(args.model)(
        pretrained=True,
        num_classes=args.nb_classes,
        intsoftmax_exp_n=args.intsoftmax_exp_n,
        intgelu_exp_n=args.intgelu_exp_n,
        attn_quant=args.attn_quant,
    )
    model.to(device)

    # counter
    cnt_linear_fc = 0
    cnt_linear_conv = 0
    cnt_linear_act = 0
    cnt_int_ln = 0
    cnt_int_gelu = 0
    cnt_int_softmax = 0
    cnt_log_act = 0
    cnt_int_mm = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            cnt_linear_fc += 1
        elif isinstance(module, QuantConv2d):
            cnt_linear_conv += 1
        elif isinstance(module, QuantAct):
            cnt_linear_act += 1
        elif isinstance(module, IntLayerNorm):
            cnt_int_ln += 1
        elif isinstance(module, IntGELU):
            cnt_int_gelu += 1
        elif isinstance(module, IntSoftmax):
            cnt_int_softmax += 1
        elif isinstance(module, Log2_half_Int_Quantizer):
            cnt_log_act += 1
        elif isinstance(module, QuantMatMul):
            cnt_int_mm += 1
    logging.info("    Number of QuantLinear: %d" % cnt_linear_fc)
    logging.info("    Number of QuantConv2d: %d" % cnt_linear_conv)
    logging.info("    Number of QuantAct: %d" % cnt_linear_act)
    logging.info("    Number of IntLayerNorm: %d" % cnt_int_ln)
    logging.info("    Number of IntGELU: %d" % cnt_int_gelu)
    logging.info("    Number of IntSoftmax: %d" % cnt_int_softmax)
    logging.info("    Number of Log2_Quantizer: %d" % cnt_log_act)
    logging.info("    - type : %s" % args.attn_quant)
    logging.info("    Number of QuantMatMul: %d" % cnt_int_mm)

    for param in model.parameters():
        param.requires_grad = False

    # calib
    unfreeze_model(model)
    model.eval()
    for i, (data, target) in enumerate(train_loader):
        if i == args.calib_images // args.calib_batchsize:
            print("calib done")
            break
        else:
            pass
        print(".", end="")

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            _ = model(data)

    freeze_model(model)
    criterion_v = nn.CrossEntropyLoss()
    _ = validate(args, val_loader, model, criterion_v, device)

    return None


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    freeze_model(model)

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(i)
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Time: %.2f" % (time.time() - start_time))
