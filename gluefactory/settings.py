from pathlib import Path

import os
if os.getenv('GLUE_SANDBOX'):
    root = Path(os.getenv('GLUE_SANDBOX'))
else:
    root = Path(__file__).parent.parent  # top-level directory

DATA_PATH = root / "data/"  # datasets and pretrained weights
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = root / "outputs/results/"  # evaluation results