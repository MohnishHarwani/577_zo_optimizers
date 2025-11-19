"""Script to benchmark optimizers and save results to a json file.

Example:

python -m celo.benchmark_runtime \
--benchmark_opt adam \
--benchmark_opt_type handcrafted \
--benchmark_task ImageMLP_Cifar100_128x128x128_Relu \
--benchmark_log_dir benchmark/

Results will be saved to benchmark/benchmark_results.json.

Checkout celo/factory.py for list of supported optimizers
and learned_optimization/tasks directory for all tasks.
"""

import functools
import json
import os
import sys
from collections import defaultdict

import numpy as onp
from absl import app, flags, logging
from learned_optimization import filesystem
from learned_optimization.tasks.base import single_task_to_family
from learned_optimization.tasks.fixed.conv import *
from learned_optimization.tasks.fixed.es_wrapped import *
from learned_optimization.tasks.fixed.image_mlp import *
from learned_optimization.tasks.fixed.image_mlp_ae import *
from learned_optimization.tasks.fixed.lopt import *
from learned_optimization.tasks.fixed.mlp_mixer import *
from learned_optimization.tasks.fixed.rnn_lm import *
from learned_optimization.tasks.fixed.transformer_lm import *
from learned_optimization.tasks.fixed.vit import *
from learned_optimization.tasks.quadratics import *
from learned_optimization.time_filter import timings

from celo.factory import get_optimizer
from celo.tasks import *

# ----------------------------------------------------------------
# Benchmarking config
# ----------------------------------------------------------------
# fmt: off
flags.DEFINE_string("benchmark_opt", None, "optimizer to benchmark")
flags.DEFINE_enum("benchmark_opt_type", 'handcrafted', ["learned", "handcrafted"], "type of optimizer which is being benchmarked")
flags.DEFINE_string("benchmark_task", "ImageMLP_Cifar100_128x128x128_Relu", "task for meta-testing")
flags.DEFINE_string("benchmark_log_dir", "benchmark/", "dir to dump benchmark results")
flags.DEFINE_integer("benchmark_num_runs", 30, "number of unrolls for time estimate, each unroll consists of 10 steps")
flags.DEFINE_integer("benchmark_num_tasks", 1, "number of tasks to run in parallel for benchmarking")
# fmt: on
flags.mark_flag_as_required("benchmark_opt")
FLAGS = flags.FLAGS


def main(_):
    num_runs = FLAGS.benchmark_num_runs
    num_tasks = FLAGS.benchmark_num_tasks
    logdir = FLAGS.benchmark_log_dir
    task_name = FLAGS.benchmark_task
    opt_name = FLAGS.benchmark_opt
    opt_type = FLAGS.benchmark_opt_type
    results = {
        "task": task_name,
        "optimizer_name": opt_name,
        "optimizer_type": opt_type,
        "num_runs": num_runs,
        "num_tasks": num_tasks,
    }
    opt = get_optimizer(opt_name)
    task = eval(f"{task_name}()")
    task_family = single_task_to_family(task)
    benchmark = functools.partial(
        timings.task_family_runtime_stats,
        task_family=task_family,
        num_time_estimates=num_runs,
        num_tasks_list=[num_tasks],
    )
    if opt_type == "learned":
        time_results = benchmark(lopt=opt)
    elif opt_type == "handcrafted":
        time_results = benchmark(opt=opt)
    else:
        raise ValueError(
            f"Invalid optimizer type: {opt_type}, valid options: ['learned', 'handcrafted']"
        )

    in_ms = onp.asarray(time_results[f"unroll_{num_tasks}x10"]) * 1000
    results["time_results"] = time_results
    results["time_per_step_ms"] = list(in_ms)[0]
    logging.info(
        f"Task={task_name} | Optimizer={opt_name} | Type={opt_type} | Time per step (ms)={in_ms[0]} +/- {in_ms[1]}"
    )

    json_string = json.dumps(results, indent=4, sort_keys=True)
    logging.info(json_string)
    filesystem.make_dirs(logdir)
    save_path = os.path.join(logdir, "benchmark_results.json")
    with open(save_path, "w") as f:
        f.write(json_string)
        logging.info(f"Saved benchmark results at: {save_path}")


if __name__ == "__main__":
    app.run(main)
