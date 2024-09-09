import argparse
import os
import time
import math
import logging
import numpy as np

import torch
import torch.nn as nn
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy

from models import *
from utils import *


parser = argparse.ArgumentParser(description="I-ViT")

parser.add_argument(
    "--model",
    default="deit_tiny",
    choices=[
        "deit_tiny",
        "deit_small",
        "deit_base",
        "swin_tiny",
        "swin_small",
        "swin_base",
    ],
    help="model",
)
parser.add_argument(
    "--data", metavar="DIR", default="data/ImageNet", help="path to dataset"
)
parser.add_argument(
    "--data-set",
    default="IMNET",
    choices=["CIFAR", "IMNET"],
    type=str,
    help="Image Net dataset path",
)
parser.add_argument("--nb-classes", default=1000, type=int, help="number of classes")
parser.add_argument("--input-size", default=224, type=int, help="images input size")
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument(
    "--output-dir",
    type=str,
    default="results/",
    help="path to save log and quantized model",
)

# parser.add_argument("--resume", default="", help="resume from checkpoint")
# parser.add_argument(
#     "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
# )
parser.add_argument("--batch-size", default=128, type=int)
# parser.add_argument("--epochs", default=90, type=int)
parser.add_argument("--num-workers", default=8, type=int)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
parser.set_defaults(pin_mem=True)


def str2model(name):
    d = {
        "deit_tiny": deit_tiny_patch16_224,
        "deit_small": deit_small_patch16_224,
        "deit_base": deit_base_patch16_224,
        # "vit_base": vit_base_patch16_224,
        # "vit_large": vit_large_patch16_224,
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
    train_loader, val_loader = dataloader(args)

    # Model
    model = str2model(args.model)(
        pretrained=True,
        num_classes=args.nb_classes,
    )
    model.to(device)
    # calib
    unfreeze_model(model)
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            _ = model(data)

        if i == 16:
            print("calib done")
            break

    freeze_model(model)
    criterion_v = nn.CrossEntropyLoss()
    _ = validate(args, val_loader, model, criterion_v, device)

    return None


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    # progress = ProgressMeter(
    #     len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    # )
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
    main()
