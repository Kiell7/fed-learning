import torch

from matplotlib import pyplot as plt

def hist(x, title):
    x = x.detach().cpu().numpy().ravel() if isinstance(x, torch.Tensor) else np.ravel(x)
    plt.hist(x, bins=1000, color='steelblue', edgecolor='black')
    plt.title(title)
    plt.show()
    plt.savefig(f"{title}.png")