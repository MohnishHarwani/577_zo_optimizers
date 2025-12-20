import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATASET_TASKS = [
    "TransformerLM_LM1B_MultiRuntime_0",
    "RNNLM_lm1bbytes_Patch32_LSTM128_Embed64",
    "RNNLM_lm1b32k_Patch32_LSTM256_Embed128",
    "RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128",
]

file = "./multi_dataset_optimizer_benchmark.csv"

for task in DATASET_TASKS:

    df = pd.read_csv(file)
    print(df.keys())


    #plotting 
    plt.figure(figsize=(6,4))

    df = df[df['dataset'] == task]


    x = np.linspace(0, 2000, 2000)

    plt.title("Optimizers Training Losses vs. Step")

    span = 50

    celo_df = df[df['optimizer'] == 'celo']
    celo_sam_df = df[df['optimizer'] == 'celo_sam']
    adam_df = df[df['optimizer'] == 'adam']
    sgd_df = df[df['optimizer'] == 'sgd']
    adafactor_df = df[df['optimizer'] == 'adafactor']

    print("------------------------------")
    print(task)
    print("------------------------------")
    print("Final train losses")
    print(f"celo: {celo_df['train_loss'].iloc[-1]}")
    print(f"celo_sam: {celo_sam_df['train_loss'].iloc[-1]}")
    print(f"adam: {adam_df['train_loss'].iloc[-1]}")
    print(f"sgd: {sgd_df['train_loss'].iloc[-1]}")
    print(f"adafactor: {adafactor_df['train_loss'].iloc[-1]}")


    plt.plot(x, celo_df['train_loss'].ewm(span=span,adjust=False).mean(), label="CeLO")
    plt.plot(x, celo_sam_df['train_loss'].ewm(span=span,adjust=False).mean(), label="CeLO-SAM")
    plt.plot(x, adam_df['train_loss'].ewm(span=span,adjust=False).mean(), label="Adam")
    plt.plot(x, sgd_df['train_loss'].ewm(span=span,adjust=False).mean(), label="SGD")
    plt.plot(x, adafactor_df['train_loss'].ewm(span=span,adjust=False).mean(), label="Adafactor")

    plt.xlabel("Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()

    plt.savefig("./main_experiment_figures/" + task + ".pdf", dpi=300, bbox_inches='tight')
