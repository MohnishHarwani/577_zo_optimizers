# print_param_counts.py

import jax
from multi_dataset_flops_benchmark import (
    DATASET_TASKS,
    make_task_and_init,
    count_params,
)

def main():
    seed = 0
    print("Trainee model parameter counts:\n")
    for dataset_name, task_ctor in DATASET_TASKS.items():
        # Reuse your existing initialization logic
        _, task, params, model_state = make_task_and_init(task_ctor, seed)
        n_params = count_params(params)
        print(f"{dataset_name}: {n_params:,} parameters")

if __name__ == "__main__":
    main()

