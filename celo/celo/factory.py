"""
Contains all the getter methods to build different
objects like task, model, optimizer, etc.
"""

import resource
from celo.optimizers.celo_cam import Celo as CeloCAM
from absl import flags

# ----------------------------------------------------------------
# Meta-training tasks
# ----------------------------------------------------------------
from learned_optimization.tasks.base import single_task_to_family
from learned_optimization.tasks.fixed import conv, image_mlp

from celo import tasks

# Fix for tfds data build
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def get_task_family(name):
    if name == "fast_velo":
        # fast meta-training tasks from velo
        task_list = [
            tasks.cifar_task(8),
            tasks.fashion_mnist_task(8),
            tasks.mnist_task(8),
            tasks.svhn_task(8),
        ]
        return [single_task_to_family(task) for task in task_list]
    elif name == "fmnist":
        task_list = [image_mlp.ImageMLP_FashionMnist_Relu128x128()]
        return [single_task_to_family(task) for task in task_list]
    elif name == "cifar10":
        task_list = [conv.Conv_Cifar10_32x64x64()]
        return [single_task_to_family(task) for task in task_list]
    elif name == "fmnist+cifar10":
        fmnist_task = image_mlp.ImageMLP_FashionMnist_Relu128x128()
        cifar10_task = conv.Conv_Cifar10_32x64x64()
        task_list = [fmnist_task, cifar10_task]
        return [single_task_to_family(task) for task in task_list]
    else:
        raise NotImplementedError(f"Can't find task family: {name}")


# ----------------------------------------------------------------
# Task augmentation
# ----------------------------------------------------------------
from learned_optimization.tasks.task_augmentation import ReparamWeightsFamily

from celo.tasks import ReparamWeightsListFamily

flags.DEFINE_list("aug_reparam_range", [0.001, 1000.0], "param scale range for reparam")
flags.DEFINE_string("aug_reparam_level", "global", "param scale range for reparam")
flags.DEFINE_list(
    "aug_reparam_levels",
    ["none", "none", "global", "global", "tensor", "tensor", "parameter"],
    "param scale range for reparam",
)
flags.DEFINE_list(
    "aug_reparam_ranges",
    [[0.001, 1000.0], [0.01, 100.0], [0.1, 10.0]],
    "param scale ranges",
)
FLAGS = flags.FLAGS


def get_augmented_task_family(name, task_family):
    if name == "reparam":
        aug_level = FLAGS.aug_reparam_level
        aug_range = FLAGS.aug_reparam_range
        aug_range = [float(i) for i in aug_range]
        return ReparamWeightsFamily(task_family, aug_level, tuple(aug_range))
    elif name == "reparam_list":
        aug_levels = FLAGS.aug_reparam_levels
        aug_ranges = FLAGS.aug_reparam_ranges
        aug_ranges = [[float(i) for i in aug_range] for aug_range in aug_ranges]
        return ReparamWeightsListFamily(task_family, aug_levels, aug_ranges)
    else:
        raise NotImplementedError(f"Can't find task augmentation: {name}")


# ----------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------
from learned_optimization.optimizers import nadamw, optax_opts

from celo.optimizers import adafac_mlp_lopt
from celo.optimizers import celo as celo_lopt
from celo.optimizers import celo_adam, nn_adam, rnn_mlp_lopt, velo

flags.DEFINE_float("test_lr", 1e-3, "learning rate, not applicable to learned optimizers")


def get_optimizer(name):
    # --- learned optimizers --- #
    if name == "celo_phase1":
        # celo phase 1
        return celo_lopt.Celo(
            lstm_hidden_size=64,
            initial_momentum_decays=(0.9, 0.99, 0.999),
            initial_rms_decays=(0.999,),
            initial_adafactor_decays=(0.9, 0.99, 0.999),
            param_inits=1,
            train_phase=1,
            summarize_each_layer=False,
        )
    elif name == "celo":
        # celo phase 2
        # this is used for celo meta-testing
        return celo_lopt.Celo(
            lstm_hidden_size=64,
            initial_momentum_decays=(0.9, 0.99, 0.999),
            initial_rms_decays=(0.999,),
            initial_adafactor_decays=(0.9, 0.99, 0.999),
            param_inits=1,
            train_phase=2,
            summarize_each_layer=False,
        )
    elif name == "celo_adam":
        return celo_adam.Celo(
            lstm_hidden_size=64,
            initial_epsilon=1e-8,
            initial_momentum_decays=(0.9,),
            initial_rms_decays=(0.999,),
        )
    elif name == "velo_4000":
        # velo pre-trained for 4000 TPU months
        return velo.HyperV2(
            use_bugged_next_lstm_state=False,
            use_bugged_loss_features=False,
            lstm_hidden_size=512,
            param_inits=256,
            ff_mult=100.0,
        )
    elif name == "velo_s":
        # small velo
        return velo.HyperV2(
            use_bugged_next_lstm_state=False,
            use_bugged_loss_features=False,
            lstm_hidden_size=64,
            param_inits=32,
            ff_mult=1.0,
        )
    elif name == "celo_cam":
        return CeloCAM(
            lstm_hidden_size=64,
            initial_momentum_decays=(0.9, 0.99, 0.999),
            initial_rms_decays=(0.999,),
            initial_adafactor_decays=(0.9, 0.99, 0.999),
            param_inits=1,
            train_phase=2,             # this is meta-test phase
            summarize_each_layer=False,
            cam_n_proj=64,             # set defaults — from Section 3.3 Rec.
            cam_r_feats=128,
            cam_tau=0.02               # small temporal decay
        )
    elif name == "velo":
        return velo.HyperV2(
            use_bugged_next_lstm_state=False,
            use_bugged_loss_features=False,
            lstm_hidden_size=512,
            param_inits=256,
            ff_mult=1.0,
        )
    elif name == "rnn_mlp":
        return rnn_mlp_lopt.RNNMLPLOpt()
    elif name == "adafac_mlp":
        return adafac_mlp_lopt.AdafacMLPLOpt()
    elif name == "nnadam":
        return nn_adam.NNAdam(lstm_hidden_size=64)

    # --- hand-crafted optimizers --- #
    elif name == "adam":
        return optax_opts.Adam(FLAGS.test_lr)
    elif name == "sgd":
        return optax_opts.SGD(FLAGS.test_lr)
    elif name == "sgdm":
        return optax_opts.SGDM(FLAGS.test_lr)
    elif name == "adamw":
        return optax_opts.AdamW(FLAGS.test_lr)
    elif name == "rmsprop":
        return optax_opts.RMSProp(FLAGS.test_lr)
    elif name == "fromage":
        return optax_opts.Fromage(FLAGS.test_lr)
    elif name == "sm3":
        return optax_opts.SM3(FLAGS.test_lr)
    elif name == "lars":
        return optax_opts.Lars(FLAGS.test_lr)
    elif name == "lamb":
        return optax_opts.Lamb(FLAGS.test_lr)
    elif name == "adabelief":
        return optax_opts.AdaBelief(FLAGS.test_lr)
    elif name == "adafactor":
        return optax_opts.Adafactor(FLAGS.test_lr)
    elif name == "adagrad":
        return optax_opts.AdaGrad(FLAGS.test_lr)
    elif name == "yogi":
        return optax_opts.Yogi(FLAGS.test_lr)
    elif name == "radam":
        return optax_opts.RAdam(FLAGS.test_lr)
    elif name == "nadamw":
        return nadamw.NAdamW(FLAGS.test_lr)
    else:
        raise NotImplementedError(f"Can't find optimizer: {name}")


from learned_optimization.outer_trainers import full_es, truncated_pes

# ----------------------------------------------------------------
# Gradient estimators
# ----------------------------------------------------------------
from learned_optimization.outer_trainers.lopt_truncated_step import VectorizedLOptTruncatedStep
from learned_optimization.outer_trainers.truncation_schedule import (
    LogUniformLengthSchedule,
    NeverEndingTruncationSchedule,
)

flags.DEFINE_integer("steps_per_jit", 10, "number of steps per jit in truncated unroll")
flags.DEFINE_integer("num_tasks", 8, "number of steps per jit in truncated unroll")
flags.DEFINE_integer("min_unroll_length", 100, "minimum unroll length during meta-training")
flags.DEFINE_integer("max_unroll_length", 2000, "maximim unroll length during meta-training")
flags.DEFINE_integer("trunc_length", 50, "truncation length in PES")


def get_grad_estimator(name, task_family, lopt):
    if name == "pes":
        return truncated_pes.TruncatedPES(
            truncated_step=VectorizedLOptTruncatedStep(
                task_family=task_family,
                learned_opt=lopt,
                trunc_sched=LogUniformLengthSchedule(
                    min_length=FLAGS.min_unroll_length,
                    max_length=FLAGS.max_unroll_length,
                ),
                num_tasks=FLAGS.num_tasks,
                random_initial_iteration_offset=FLAGS.max_unroll_length,
            ),
            trunc_length=FLAGS.trunc_length,
            steps_per_jit=FLAGS.steps_per_jit,
        )

    elif name == "full_es":
        # modified from velo to match PES specs at small scale
        return full_es.FullES(
            truncated_step=VectorizedLOptTruncatedStep(
                task_family=task_family,
                learned_opt=lopt,
                trunc_sched=NeverEndingTruncationSchedule(),
                num_tasks=FLAGS.num_tasks,
                random_initial_iteration_offset=2000,
            ),
            truncation_schedule=LogUniformLengthSchedule(
                min_length=FLAGS.min_unroll_length,
                max_length=FLAGS.max_unroll_length,
            ),
            loss_type="last_recompute",
            sign_delta_loss_scalar=1.0,
        )
    elif name == "full_es_velo":
        # config from velo
        return full_es.FullES(
            truncated_step=VectorizedLOptTruncatedStep(
                task_family=task_family,
                learned_opt=lopt,
                trunc_sched=NeverEndingTruncationSchedule(),
                num_tasks=8,
                random_initial_iteration_offset=10000,
            ),
            truncation_schedule=LogUniformLengthSchedule(min_length=100, max_length=10000),
            loss_type="last_recompute",
            sign_delta_loss_scalar=1.0,
        )
    else:
        raise NotImplementedError(f"Can't find gradient estimator: {name}")
