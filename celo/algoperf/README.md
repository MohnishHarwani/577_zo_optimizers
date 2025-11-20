# AlgoPerf benchmark

[MLCommons AlgoPerf](https://github.com/mlcommons/algorithmic-efficiency) is a time-to-target benchmark that evaluates optimizers on 8 standard tasks including image classification, language modeling, speech recognition modeling, etc. Each task has a pre-defined valiation target which should be obtained by the optimizer. The full list of tasks along with their targets is available [here](https://github.com/mlcommons/algorithmic-efficiency/blob/main/docs/DOCUMENTATION.md#fixed-workloads).

The benchmark has two tracks: (1) Self-tuning (3) External tuning. External tuning track is mainly for hand-crafted optimizers (such as SGD, Adam) that have tunable hyperparameters. Hyperparam search can be supplied as a json file along with optimizer code for this track. Since our focus is learned optimizers, we use self-tuning track which doesn't require any tuning hyperparams for each task. Hence, we only provide the submission file that can be used directly to evaluate learned optimizers on this benchmark.

## Setup datasets

Set `$DATA_DIR` path:
```
export DATA_DIR="/path/to/algoperf_datasets"
```

Set resource limits (for ImageNet dataset setup):
```
ulimit -n 8192
```

Then, follow [instructions](https://github.com/mlcommons/algorithmic-efficiency/tree/c96ba6269f71f1c7bb2e5724f8cab82a35ab6580/datasets#individual-dataset-instructions) from the official AlgoPerf repo to setup all the datasets.

## Run

We provide submission files for Celo, Celo-Adam (variant of Celo that uses Adam update rule instead of learned MLP update) and VeLO as example scripts for AlgoPerf evaluation in `submissions/` directory. All the submission scripts are roughly the same except `init_optimizer_state` method that loads and builds pretrained optimizer object. Other (or new!) learned optimizers can be evaluated by simply modifying the `init_optimizer_state` method in any of these submission files while keeping rest of the code as it is. Make sure that the learned optimizer checkpoint is correct and located in the corresponding submission directory.

To run a learned optimizer on AlgoPerf, first clone [algorithmic-efficiency](https://github.com/mlcommons/algorithmic-efficiency/) and copy the respective optimizer directory (say `submissions/celo_adam`) in algorithmic-efficiency `submissions` directory:
```
git clone git@github.com:mlcommons/algorithmic-efficiency.git
cd algorithmic-efficiency
cp -r /path/to/celo/algoperf/submissions/celo_adam submissions/
```

We provide example commands below for each task in AlgoPerf to test learned optimizers. `max_global_steps` argument for each workload is picked from the prize qualification baseline which was used for tuning. In AlgoPerf self-tuning competition track, 3x of this value is used to give more time to self-tuning algorithms to hit the targets.

The commands below should be run from algorithmic-efficiency directory and following flags should be modified as per the use case:
- `--submission_path` -- path to optimizer being evaluated
- `--experiment_dir` -- meta directory for saving run artifacts for all experiments
- `--experiment_name` -- name of run directory which will be created in the specified `--experiment_dir`
- `--overwrite` -- if true, any artifacts that already exist in specified experiment directory will be overwritten

Checkout full list of flags [here](https://github.com/mlcommons/algorithmic-efficiency/blob/c96ba6269f71f1c7bb2e5724f8cab82a35ab6580/submission_runner.py#L64-L165).

### ImageNet ResNet50

```
python submission_runner.py --framework=jax --tuning_ruleset=self --nosave_checkpoints --experiment_dir=algoperf_exps/ --experiment_name=celo_adam --submission_path=submissions/celo_adam/submission.py --workload=imagenet_resnet --data_dir=$DATA_DIR/imagenet/jax --max_global_steps=186666 --rng_seed=0 --overwrite
```

### ImageNet ViT

```
python submission_runner.py --framework=jax --tuning_ruleset=self --nosave_checkpoints --experiment_dir=algoperf_exps/ --experiment_name=celo_adam --submission_path=submissions/celo_adam/submission.py --workload=imagenet_vit --data_dir=$DATA_DIR/imagenet/jax --max_global_steps=186666 --rng_seed=0 --overwrite
```

### Criteo1TB

```
python submission_runner.py --framework=jax --tuning_ruleset=self --nosave_checkpoints --experiment_dir=algoperf_exps/ --experiment_name=celo_adam --submission_path=submissions/celo_adam/submission.py --workload=criteo1tb --data_dir=$DATA_DIR/criteo1tb --max_global_steps=10666 --rng_seed=0 --overwrite
```

### Fast MRI

```
python submission_runner.py --framework=jax --tuning_ruleset=self --nosave_checkpoints --experiment_dir=algoperf_exps/ --experiment_name=celo_adam --submission_path=submissions/celo_adam/submission.py --workload=fastmri --data_dir=$DATA_DIR/fastmri --max_global_steps=36189 --rng_seed=0 --overwrite
```

### LibriSpeech Conformer

```
python submission_runner.py --framework=jax --tuning_ruleset=self --nosave_checkpoints --experiment_dir=algoperf_exps/ --experiment_name=celo_adam --submission_path=submissions/celo_adam/submission.py --workload=librispeech_conformer --data_dir=$DATA_DIR/librispeech --librispeech_tokenizer_vocab_path=$DATA_DIR/librispeech/spm_model.vocab  --max_global_steps=80000 --rng_seed=0 --overwrite
```

### OGBG

```
python submission_runner.py --framework=jax --tuning_ruleset=self --nosave_checkpoints --experiment_dir=algoperf_exps/ --experiment_name=celo_adam --submission_path=submissions/celo_adam/submission.py --workload=ogbg --data_dir=$DATA_DIR/ogbg --max_global_steps=80000 --rng_seed=0 --overwrite
```


### WMT

```
python submission_runner.py --framework=jax --tuning_ruleset=self --nosave_checkpoints --experiment_dir=algoperf_exps/ --experiment_name=celo_adam --submission_path=submissions/celo_adam/submission.py --workload=wmt --data_dir=$DATA_DIR/wmt --max_global_steps=133333 --rng_seed=0 --overwrite
```

## Preliminary results

Note that the tasks in this benchmark are larger and much more compute-intensive than the ones considered in our work (which were limited to <1h evaluation time per task on 1 GPU for thorough ablations): some of these AlgoPerf tasks may take 2-3 days to finish on a 8 GPU machine. Due to compute-intensive nature of this benchmark, it poses a substantial challenge especially for learned optimizers which often require considerable resources for meta-training with long unroll lengths to perform well on these tasks.

Although this benchmark was not the focus in our work, we provide scripts for evaluating learned optimizers on AlgoPerf to support future research that may build upon our approach to further improve compute-efficiency and meta-generalization of learned optimizers. Preliminary results on the AlgoPerf benchmark are reported in Appendix section A.6 of our [paper](https://arxiv.org/abs/2501.12670) that directly test learned optimizers pretrained on small image MLP tasks. In short, our variant of Celo, namely, Celo-Adam (learned scheduler with Adam update rule, also discussed in Table 5), manages to optimize relatively stably and even better in ResNet tasks than Schedule-Free AdamW (winner of self-tuning track 2024). However, the final validation performance of Celo-Adam still lags behind Schedule-Free AdamW and sometimes fails to hit validation targets earlier as evident from the validation curves. We find that Celo and other learned optimizer baselines were unstable on these AlgoPerf tasks potentially due to their limited meta-training on small image MLP tasks with much shorter meta-training horizon (2K).

If you use or find these scripts helpful, please cite our work:
```
@article{moudgil2025celo,
  title={Celo: Training Versatile Learned Optimizers on a Compute Diet},
  author={Abhinav Moudgil and Boris Knyazev and Guillaume Lajoie and Eugene Belilovsky},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=SLqJbt4emY}
}
```

as well as AlgoPerf benchmark:
```
@Misc{Dahl2023AlgoPerf,
  title         = {{Benchmarking Neural Network Training Algorithms}},
  author        = {Dahl, George E. and Schneider, Frank and Nado, Zachary and Agarwal, Naman and Sastry, Chandramouli Shama and Hennig, Philipp and Medapati, Sourabh and Eschenhagen, Runa and Kasimbeg, Priya and Suo, Daniel and Bae, Juhan and Gilmer, Justin and Peirson, Abel L. and Khan, Bilal and Anil, Rohan and Rabbat, Mike and Krishnan, Shankar and Snider, Daniel and Amid, Ehsan and Chen, Kongtao and Maddison, Chris J. and Vasudev, Rakshith and Badura, Michal and Garg, Ankush and Mattson, Peter},
  year          = {2023},
  archiveprefix = {arXiv},
  eprint        = {2306.07179},
}
```
