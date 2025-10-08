import os
import random
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

DEBUG_PRINT = True # whether to print verbose debugging output

# Identifiers for dictionary key values
# For communication information
ACTIVATION_SIZE = 'activation_size' 
GRADIENT_SIZE = 'gradient_size'

# For per-rank statistics
RANK = 'rank'
TRAIN_LOSS = 'train_loss'
LR = 'learning_rate'

# For overall statistics
CORES_PER_WORKER = 'cores_per_worker'
NUM_WORKERS = 'num_workers'
NUM_BATCHES = 'num_batches'
BATCH_SIZE = 'batch_size'

def debug_print(*args, **kwargs):
    """
    Togglable printing, according to the `DEBUG_PRINT` flag in `utils.py`.
    """
    if DEBUG_PRINT:
        print(*args, **kwargs)

def clear_dir(path: str):
    """
    Clears the given directory of all saved PyTorch files.
    
    Args:
        path (str): the path of the given directory to clear
    """
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if filename.endswith(".pt"):
            os.remove(filepath)

def seed_everything(s: int):
    """
    This function allows us to set the seed for all of our random functions
    so that we can get reproducible results.

    Args:
        s (int): the seed to seed all random functions with
    """
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def agg_stats_per_rank(stats_queue: mp.Queue) -> dict[int, dict]:
    """
    Aggregates the stats from all ranks, passed via the given queue, into
    a single dictionary.
    
    Args:
        stats_queue (mp.Queue): the queue used to communicate each rank's 
        statistics to the main process
    Returns:
        dict[int, dict]: a dictionary from each rank to the rank's statistics
    """
    stats = {}
    while True:
        try:
            rank_stats = stats_queue.get(block=False)
        except queue.Empty:
            break
        stats[rank_stats['rank']] = rank_stats
    return stats


# ---------------------------------------------------------------------------------
# Parallelism training helpers

def parallel_setup(rank: int, world_size: int):
    """
    Performs any generic set-up for parallel training for the current process.
    
    Args:
        rank (int): the rank of the current process
        world_size (int): the total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.autograd.set_detect_anomaly(True)

def pin_to_core(rank: int, num_cores: int):
    """
    Pins the current process to a specific number of unique cores.
    
    Args:
        rank (int): the rank of the current process
        num_cores (int): the number of cores to pin the current process to
    """
    start_core = rank * num_cores
    end_core = start_core + num_cores
    cores_print = f"core{f' {start_core}' if num_cores == 1 else f's {start_core}-{end_core-1}'}"
    debug_print(f"Rank {rank} pinning to {cores_print}")
    os.sched_setaffinity(0, range(start_core, end_core))
    debug_print(f"Rank {rank} pinned to {cores_print}")

def parallel_cleanup():
    """
    Performs any generic cleaning up for parallel training for the current process.
    ALL work for the process should be performed before this method is called.
    """
    dist.destroy_process_group()