"""
Author: Paul-Edouard Sarlin (skydes)
Modder: Hoang-QC
"""

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


from gluefactory import __module_name__, logger
from gluefactory.datasets import get_dataset
from gluefactory.settings import EVAL_PATH, TRAINING_PATH
from gluefactory.utils.image import read_image
from gluefactory.geometry.homography import (
    compute_homography,
    sample_homography_corners,
    warp_points,
)

from gluefactory.models import get_model


from matplotlib import pyplot as plt

# dumb_exp --conf=configs/superpoint+lightglue_homography_debugging.yaml
import sys
if "pydevconsole.py" in sys.argv[0]:
    print("Pycharm Console mode!")
    sys.argv=['/home/hoangqc/Sfm/glue-factory/gluefactory/validation_pipeline.py',
              'dumb_exp',
              '--conf=gluefactory/configs/superpoint+lightglue_homography_debugging.yaml']
else:
    print("Normal mode!")
    sys.argv.append('dump_exp')
    sys.argv.append('--conf=configs/superpoint+lightglue_homography_debugging.yaml')

default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
        "options": {},  # add lr_scheduler arguments here
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 10,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
}
default_train_conf = OmegaConf.create(default_train_conf)

parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str)
parser.add_argument("--conf", type=str)
parser.add_argument(
    "--mixed_precision",
    "--mp",
    default=None,
    type=str,
    choices=["float16", "bfloat16"],
)

parser.add_argument(
    "--compile",
    default=None,
    type=str,
    choices=["default", "reduce-overhead", "max-autotune"],
)
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--restore", action="store_true")
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--print_arch", "--pa", action="store_true")
parser.add_argument("--detect_anomaly", "--da", action="store_true")
parser.add_argument("--log_it", "--log_it", action="store_true")
parser.add_argument("--no_eval_0", action="store_true")
parser.add_argument("--run_benchmarks", action="store_true")
parser.add_argument("dotlist", nargs="*")
args = parser.parse_intermixed_args()

logger.info(f"Starting experiment {args.experiment}")
output_dir = Path(TRAINING_PATH, args.experiment)
output_dir.mkdir(exist_ok=True, parents=True)

# conf = OmegaConf.from_cli(args.dotlist)
conf=OmegaConf.load(args.conf)

OmegaConf.save(conf, str(output_dir / "config.yaml"))
if args.conf:
    conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
elif args.restore:
    restore_conf = OmegaConf.load(output_dir / "config.yaml")
    conf = OmegaConf.merge(restore_conf, conf)
if not args.restore:
    if conf.train.seed is None:
        conf.train.seed = torch.initial_seed() & (2 ** 32 - 1)
    OmegaConf.save(conf, str(output_dir / "config.yaml"))

# copy gluefactory and submodule into output dir
for module in conf.train.get("submodules", []) + [__module_name__]:
    mod_dir = Path(__import__(str(module)).__file__).parent
    shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)

# main_worker(0, conf, output_dir, args)
# training(0, conf, output_dir, args)

conf.train = OmegaConf.merge(default_train_conf, conf.train)
data_conf = copy.deepcopy(conf.data)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device {device}")

dataset = get_dataset(data_conf.name)(data_conf)

# Optionally load a different validation dataset than the training one
val_data_conf = conf.get("data_val", None)
val_dataset = dataset

train_loader = dataset.get_data_loader("train", distributed=args.distributed)
homo_dataset = train_loader.dataset
# val_loader = val_dataset.get_data_loader("val")

# imgs = next(iter(train_loader))

# idx = 56014

# imgs = train_loader.dataset.getitem(56014)
# data0 = imgs['view0']
# data1 = imgs['view1']

# raw_img = read_image(train_loader.dataset.image_dir / imgs['name'], False)

img_name = '806/8061785ce4cc7e30d72de131fcfb42.jpg'
raw_img = read_image(homo_dataset.image_dir / img_name, False)
plt.imshow(raw_img);plt.show()

img = raw_img.astype(np.float32) / 255.0
size = img.shape[:2][::-1]

ps = homo_dataset.conf.homography.patch_shape
left_conf = OmegaConf.to_container(homo_dataset.conf.homography)

def convertRotation(cx: float, cy: float, H:  np.array):
    ct = np.array([cx,cy,1])
    p = np.dot(H, ct)
    pw = np.array([p[0]/p[2], p[1]/p[2]]) # Warp Center
    rx = - (pw[0] - cx) / cx # Right -> +
    ry = (pw[1] - cy) / cy # Up -> +
    return np.array([rx,ry]), pw

data0 = homo_dataset._read_view(img, left_conf, ps, left=True)
data0_ext = homo_dataset._read_view(img, left_conf, ps, left=True)
data1 = homo_dataset._read_view(img, homo_dataset.conf.homography, ps, left=False)
data1_ext = homo_dataset._read_view(img, homo_dataset.conf.homography, ps, left=False)

H_0to1 = compute_homography(data0["coords"], data1["coords"], [1, 1])
H_ext0 = compute_homography(data0["coords"], data0_ext["coords"], [1, 1])
H_ext1 = compute_homography(data1["coords"], data1_ext["coords"], [1, 1])

cx = ps[0]/2
cy = ps[1]/2
rw0, pw0 = convertRotation(cx,cy,H_ext0)
rw1, pw1 = convertRotation(cx,cy,H_ext1)

# center_point = torch.tensor([[320,240]])
# point_ext0 = warp_points(center_point,H_ext0,inverse=False)
# point_ext1 = warp_points(center_point,H_ext1,inverse=False)

img0 = data0['image'].permute(1, 2, 0).numpy()
img0_ext = data0_ext['image'].permute(1, 2, 0).numpy()
img1 = data1['image'].permute(1, 2, 0).numpy()
img1_ext = data1_ext['image'].permute(1, 2, 0).numpy()

import cv2 as cv
img0_x = cv.circle(img0,center=(320, 240), radius=5, color=(1,0,0),thickness=3)
img0_ext_x = cv.circle(img0_ext,center=(int(pw0[0]), int(pw0[1])), radius=5, color=(1,0,0),thickness=3)

img1_x = cv.circle(img1,center=(320, 240), radius=5, color=(1,1,0),thickness=3)
img1_ext_x = cv.circle(img1_ext,center=(int(pw1[0]), int(pw1[1])), radius=5, color=(1,1,0),thickness=3)

plt.imshow(img0_x);plt.show()
plt.imshow(img0_ext_x);plt.show()
plt.imshow(img1_x);plt.show()
plt.imshow(img1_ext_x);plt.show()

model = get_model(conf.model.name)(conf.model).to(device)
model.__f