dependencies = ['torch', 'torchvision', 'pycocotools']

import os
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

def faster_rcnn(config_path=None, weights_path=None):
    """
    Builds and returns a Faster R-CNN model using Detectron2.
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(weights_path)
    return model.eval()

def list_models():
    """
    Lists the available models in this hubconf file.
    """
    return ["faster_rcnn"]

def _download(url, dst):
    os.system(f'curl -L {url} > {dst}')

def _get_config_file(name):
    cfg_file = Path(__file__).parent / f'{name}.yaml'
    if not cfg_file.is_file():
        url = f'https://raw.githubusercontent.com/username/repo/master/models/{name}.yaml'
        _download(url, cfg_file)
    return str(cfg_file)

def _get_weights_file(name):
    weights_file = Path(__file__).parent / f'{name}.pth'
    if not weights_file.is_file():
        url = f'https://github.com/username/repo/releases/download/v0.1/{name}.pth'
        _download(url, weights_file)
    return str(weights_file)
