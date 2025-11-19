"""
Script to meta-train learned optimizers.
"""

import json
import os
import pickle
import shutil

import jax
import numpy as np
import tqdm
from absl import app, flags, logging
from learned_optimization import checkpoints, filesystem, summary, tree_utils
from learned_optimization.optimizers import optax_opts

from celo import gradient_learner
from celo.factory import (
    get_augmented_task_family,
    get_grad_estimator,
    get_optimizer,
    get_task_family,
)
from celo.utils import load_from_pretrained

# ----------------------------------------------------------------
# Meta-training config
# ----------------------------------------------------------------
jax.config.parse_flags_with_absl()
# fmt: off
flags.DEFINE_string("optimizer", None, "learned optimizer to meta-train")
flags.DEFINE_string("task", None, "task for meta-training")
flags.DEFINE_string("trainer", None, "gradient estimator for meta-training")
flags.DEFINE_string("ckpt_save_dir", "checkpoints/", "common directory for all run checkpoints")
flags.DEFINE_string("train_log_dir", "logs/train", "common directory for all run tb logs")
flags.DEFINE_string("init_from_ckpt", None, "path to serialized lopt checkpoint for initialization.")
flags.DEFINE_string("init_ckpt_optimizer_name", None, "checkpoint optimizer name to partially load params, skip if optimizer pytree matches checkpoint pytree.")
flags.DEFINE_bool("resume", False, "resume training from latest checkpoint using trainer saved state")
flags.DEFINE_bool("train_partial", False, "enables training subset of celo params while keeping rest frozen")
flags.DEFINE_string("name", None, "run name, ckpt and wandb runs are created with this name")
flags.DEFINE_float("outer_lr", 1e-4, "learning rate of meta-trainer")
flags.DEFINE_integer("outer_iterations", 100000, "number of meta-iterations")
flags.DEFINE_integer("ckpt_interval", 5000, "checkpoint save interval")
flags.DEFINE_integer("metrics_every", 10, "log interval of additional metrics and intermediate values in jit functions")
flags.DEFINE_integer("seed", 0, "training seed")
flags.DEFINE_string("aug", None, "task augmentation")
flags.DEFINE_bool("disable_wandb", False, "disable wandb logging e.g. for debugging")
flags.DEFINE_string("wandb_project", None, "wandb project name")
flags.DEFINE_string("wandb_entity", None, "wandb project entity, specify username for individuals")
# fmt: on

flags.mark_flag_as_required("optimizer")
flags.mark_flag_as_required("task")
flags.mark_flag_as_required("trainer")
flags.mark_flag_as_required("name")
FLAGS = flags.FLAGS


# ----------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------
def train(unused_argv):
    # setup logdir
    seed = FLAGS.seed
    key = jax.random.PRNGKey(seed)
    ckpt_dir = os.path.join(FLAGS.ckpt_save_dir, FLAGS.name)
    filesystem.make_dirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(FLAGS.flag_values_dict(), f, ensure_ascii=False, indent=4)
        logging.info(json.dumps(FLAGS.flag_values_dict(), ensure_ascii=False, indent=4))
        logging.info("Saved config at: {}".format(os.path.join(ckpt_dir, "config.json")))
    if FLAGS.disable_wandb:
        summary_writer = summary.PrintWriter()
    else:
        wandb_writer = summary.WandbWriter(project=FLAGS.wandb_project,
                                           entity=FLAGS.wandb_entity,
                                           logdir=FLAGS.train_log_dir,
                                           config=FLAGS.flag_values_dict(),
                                           name=FLAGS.name)
        summary_writer = summary.MultiWriter(wandb_writer, summary.PrintWriter())

    # setup optimizer and tasks for meta-training
    lopt = get_optimizer(FLAGS.optimizer)
    task_families = get_task_family(FLAGS.task)
    if FLAGS.aug:
        task_families = [get_augmented_task_family(FLAGS.aug, family) for family in task_families]

    # setup outer trainer
    gradient_estimators = []
    for task_family in task_families:
        grad_est = get_grad_estimator(FLAGS.trainer, task_family, lopt)
        gradient_estimators.append(grad_est)
    theta_opt = optax_opts.AdamW(FLAGS.outer_lr)
    frozen_keys = lopt.get_frozen_param_keys() if FLAGS.train_partial else None
    logging.info(f"Frozen keys: {frozen_keys}")
    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        lopt, gradient_estimators, theta_opt, frozen_keys
    )

    # initialize learned optimizer params from checkpoint if specified
    if FLAGS.init_from_ckpt:
        ckpt_path = os.path.join(FLAGS.ckpt_save_dir, FLAGS.init_from_ckpt)
        logging.info(f"Loading params from checkpoint: {ckpt_path}")
        # key1, key = jax x.random.split(key)
        lopt_tree = lopt.init(key)
        if FLAGS.init_ckpt_optimizer_name:
            ckpt_lopt = get_optimizer(FLAGS.init_ckpt_optimizer_name)
            # key1, key = jax.random.split(key)
            ckpt_tree = ckpt_lopt.init(key)
        else:
            ckpt_tree = lopt_tree
        pretrained_tree = checkpoints.load_state(ckpt_path, ckpt_tree)
        init_tree = load_from_pretrained(lopt_tree, pretrained_tree)
        outer_trainer.gradient_learner._init_theta = init_tree
    outer_trainer_state = outer_trainer.init(key)

    # initialize meta-training
    start_iteration = 0
    losses = []
    m = []
    m_xs = []

    # resume from checkpoint if specified
    if FLAGS.resume:
        ckpt_path = os.path.join(ckpt_dir, "trainer_latest.pkl")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        outer_trainer_state = ckpt["state"]
        m = ckpt["metrics"]
        m_xs = ckpt["metrics_xs"]
        losses = ckpt["losses"]
        start_iteration = (
            outer_trainer_state.gradient_learner_state.theta_opt_state.iteration.item()
        )
        if start_iteration == FLAGS.outer_iterations:
            logging.info(
                "Resumed job has already completed training, change `--outer_iterations` flag to train for longer"
            )
            return
    summary_writer.scalar("seed", seed, step=start_iteration)

    # meta-training loop
    for i in tqdm.trange(
        start_iteration,
        FLAGS.outer_iterations,
        initial=start_iteration,
        total=FLAGS.outer_iterations,
    ):
        with_m = True if i % FLAGS.metrics_every == 0 else False
        key1, key = jax.random.split(key)
        outer_trainer_state, loss, metrics = outer_trainer.update(
            outer_trainer_state, key1, with_metrics=with_m
        )
        losses.append(loss)

        # log summaries to wandb
        if with_m:
            log_m = {}
            summary_writer.scalar("average_meta_loss", np.mean(losses), step=i)
            log_m["average_meta_loss"] = np.mean(losses)
            for k, v in metrics.items():
                agg_type, metric_name = k.split("||")
                if agg_type == "collect":
                    summary_writer.histogram(metric_name, v, step=i)
                else:
                    summary_writer.scalar(metric_name, v, step=i)
                    log_m[metric_name] = v
            summary_writer.flush()
            m.append(log_m)
            m_xs.append(i)
            losses = []

        # save learned optimizer params periodically
        if i % FLAGS.ckpt_interval == 0:
            theta = outer_trainer.get_meta_params(outer_trainer_state)
            checkpoints.save_state(os.path.join(ckpt_dir, f"theta_{i}.state"), theta)
            trainer_state = {
                "state": outer_trainer_state,
                "metrics": m,
                "metrics_xs": m_xs,
                "losses": losses,
            }
            with open(os.path.join(ckpt_dir, "trainer_latest.pkl"), "wb") as f:
                pickle.dump(trainer_state, f)

            # save logs
            stacked_metrics = tree_utils.tree_zip_onp(m)
            ret = {f"train/{k}": v for k, v in stacked_metrics.items()}
            ret["train/xs"] = np.asarray(m_xs)
            path = os.path.join(ckpt_dir, "logs.pkl")
            with open(path, "wb") as f:
                pickle.dump(ret, f)

    # save final learned state
    path = os.path.join(ckpt_dir, "theta.state")
    theta = outer_trainer.get_meta_params(outer_trainer_state)
    checkpoints.save_state(path, theta)
    logging.info(f"Saved learned optimizer params at: {path}")
    trainer_state = {
        "state": outer_trainer_state,
        "metrics": m,
        "metrics_xs": m_xs,
        "losses": losses,
    }
    trainer_path = os.path.join(ckpt_dir, "trainer_latest.pkl")
    with open(trainer_path, "wb") as f:
        pickle.dump(trainer_state, f)
    logging.info(f"Saved trainer state at: {trainer_path}")

    # save logs
    stacked_metrics = tree_utils.tree_zip_onp(m)
    ret = {f"train/{k}": v for k, v in stacked_metrics.items()}
    ret["train/xs"] = np.asarray(m_xs)
    path = os.path.join(ckpt_dir, "logs.pkl")
    with open(path, "wb") as f:
        pickle.dump(ret, f)
    logging.info(f"Saved training logs pickle at: {path}")
    logging.info("Finished training.")


if __name__ == "__main__":
    app.run(train)
