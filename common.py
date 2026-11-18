import hashlib
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
from torch.utils.data import DataLoader, SequentialSampler

import cross_dataset

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

WORK_ROOT = Path(__file__).resolve().parent
UPT_ROOT = WORK_ROOT / "repos" / "upt"
sys.path.insert(0, str(UPT_ROOT))
os.chdir(UPT_ROOT)

_resnet50 = torchvision.models.resnet50


def _resnet50_without_download(*args, **kwargs):
    kwargs.pop("pretrained", None)
    kwargs["weights"] = None
    return _resnet50(*args, **kwargs)


torchvision.models.resnet50 = _resnet50_without_download

from upt import build_detector  # noqa: E402
from utils import CustomisedDLE, DataFactory, custom_collate  # noqa: E402

MODEL_ARGS = dict(
    backbone="resnet50", dilation=False, position_embedding="sine",
    num_queries=100, aux_loss=True, set_cost_class=1, set_cost_bbox=5,
    set_cost_giou=2, bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1,
    hidden_dim=256, enc_layers=6, dec_layers=6, dim_feedforward=2048,
    dropout=0.1, nheads=8, pre_norm=False, repr_dim=512, human_idx=0,
    alpha=0.5, gamma=0.2, box_score_thresh=0.2, fg_iou_thresh=0.5,
    min_instances=3, max_instances=15, device="cuda",
)


def resolve(path):
    return (WORK_ROOT / path).resolve()


def model_args(num_classes, pretrained=""):
    return SimpleNamespace(**MODEL_ARGS, num_classes=num_classes, pretrained=pretrained)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment():
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "gpu": torch.cuda.get_device_name(0),
    }


def cuda_peak_memory():
    mib = 1024 ** 2
    return {
        "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated(0) / mib,
        "peak_cuda_reserved_MiB": torch.cuda.max_memory_reserved(0) / mib,
    }


class ModelAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, images):
        return self.module(images)


def load_eval_model(checkpoint, source, target):
    names, class_corr = cross_dataset.classes(source.dataset)
    model = build_detector(model_args(len(names)), class_corr)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model = model.cuda()
    if source is target:
        return ModelAdapter(model).eval()
    return cross_dataset.Relabel(model, source.dataset, target.dataset).cuda().eval()


def test_loader(dataset, num_workers):
    return DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                      num_workers=num_workers, collate_fn=custom_collate,
                      pin_memory=True, drop_last=False)


def engine(model):
    return SimpleNamespace(_state=SimpleNamespace(net=model))
