import matplotlib.pyplot as plt
import pandas as pd

file = "./celo_compression_sweep.csv"

df = pd.read_csv(file)
print(df.keys())



#plotting 
plt.figure()

df = df[df['step'] < 2000]

x = df['step']

plt.title("Optimizers Training Losses vs. Step")

span = 20

celo = df[df["variant"] == "celo"].sort_values("step")
ewm_loss = celo['train_loss'].ewm(span=span, adjust=False).mean()

plt.plot(celo['step'], ewm_loss, label="Base CeLO")
plt.show()
exit()



plt.plot(x, df['celo_prune_train'].ewm(span=span,adjust=False).mean(), label="Pruned CeLO")
plt.plot(x, df['celo_q8_train'].ewm(span=span,adjust=False).mean(), label="Int8 Quantization")

plt.plot(x, df['celo2_train'].ewm(span=span,adjust=False).mean(), label="SAM Base CeLO")
plt.plot(x, df['celo2_prune_train'].ewm(span=span,adjust=False).mean(), label="SAM Pruned CeLO")
plt.plot(x, df['celo2_q8_train'].ewm(span=span,adjust=False).mean(), label="SAM Int8 Quantization")
plt.plot(x, df['adam_train'].ewm(span=span,adjust=False).mean(), label="Adam")
plt.plot(x, df['adafactor_train'].ewm(span=span,adjust=False).mean(), label="Adafactor")
plt.plot(x, df['sgd_train'].ewm(span=span,adjust=False).mean(), label="SGD")

plt.xlabel("Steps")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.savefig("training.pdf", dpi=300)
