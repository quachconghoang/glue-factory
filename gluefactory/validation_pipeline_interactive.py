"""
A generic training script that works with any model and dataset.

Author: Paul-Edouard Sarlin (skydes)
Modder: Hoang-QC
"""

import argparse
import copy
import re
import shutil
import signal
from collections import defaultdict
from pathlib import Path
from pydoc import locate

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gluefactory import __module_name__, logger
from gluefactory.datasets import get_dataset
from gluefactory.eval import run_benchmark
from gluefactory.models import get_model
from gluefactory.settings import EVAL_PATH, TRAINING_PATH
from gluefactory.utils.experiments import get_best_checkpoint, get_last_checkpoint, save_experiment
from gluefactory.utils.stdout_capturing import capture_outputs
from gluefactory.utils.tensor import batch_to_device
from gluefactory.utils.tools import (
    AverageMetric,
    MedianMetric,
    PRMetric,
    RecallMetric,
    fork_rng,
    set_seed,
)

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
# val_loader = val_dataset.get_data_loader("val")

imgs = next(iter(train_loader))