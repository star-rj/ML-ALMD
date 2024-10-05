import os

from mmdet.datasets.builder import build_dataset
from mmcv import Config, DictAction

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config_path = './configs/M/cascade_rcnn_r101_M_17.py'
# config_path = './configs/M/tmp.py'
cfg = Config.fromfile(config_path)
dataset = build_dataset(cfg.data.test)

for item in dataset:
    pass
