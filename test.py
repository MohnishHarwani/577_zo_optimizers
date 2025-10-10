
import torch
print(torch.__version__)
print(torch.version.cuda)   # CUDA version PyTorch was built with
print(torch.backends.cudnn.version())  # optional, cuDNN version
