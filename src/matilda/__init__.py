"""matilda-sc — multi-task framework for single-cell multimodal data.

Joint cell-type classification, dimension reduction, feature selection, data
simulation, and augmentation across RNA, ADT (protein), and ATAC modalities.

The import name is ``matilda``; the PyPI distribution is ``matilda-sc``.

Importing this package does no argument parsing and runs no training. (It does bind
torch tensor types to the visible device at import time, via ``util.py``.) The
command-line entry points live behind ``if __name__ == "__main__"`` guards in each
``main_matilda_*`` module.

Quickstart (path-based API, model unchanged from the published engine)::

    from matilda import main_train, main_task
    main_train("rna.h5", "adt.h5", "atac.h5", "cty.csv", seed=1)
    main_task("rna.h5", "adt.h5", "atac.h5", "cty.csv",
              classification=True, query=True, seed=1)

A higher-level object API (``matilda.train`` / ``matilda.task``) that takes
AnnData / arrays and returns result objects is provided by ``matilda.api``
(I/O staging in ``matilda.io``); see the recommended entry points below.
"""

from .main_matilda_train import main_train
from .main_matilda_task import main_task
from .main_matilda_rna_train import rna_train
from .main_matilda_rna_task import rna_task

from .api import (train, task, classify, reduce, markers, simulate,
                  TrainResult, TaskResult)

__version__ = "0.2.0"

__all__ = [
    # object API (recommended)
    "train",
    "task",
    "classify",
    "reduce",
    "markers",
    "simulate",
    "TrainResult",
    "TaskResult",
    # raw path-based engine functions
    "main_train",
    "main_task",
    "rna_train",
    "rna_task",
]
