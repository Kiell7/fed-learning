import torch

def backslash(model, gamma_table, r_gamma_table, rdo=1e5):
    with torch.no_grad():
        device = torch.device("cuda:0")
        n, var, mean = 0, 0, 0
        for param in model.parameters():
            param = param.flatten().detach()
            n += param.shape[0]
            var += torch.sum((param ** 2).to(device))
            mean += torch.sum(torch.abs(param).to(device))

        r_gamma = (n * var / mean ** 2).to(device=torch.device("cpu"))
        pos = torch.argmin(torch.abs(r_gamma - r_gamma_table))
        shape = gamma_table[pos]
        std = torch.sqrt(var / n)
        n = torch.tensor(n)

        for param in model.parameters():
            constant = rdo * shape / n * torch.sign(param.data)
            param_reg = torch.pow(
                torch.abs(param.data) + 1e-5, shape - 1)
            param.data -= constant * param_reg
        distribution = {"shape": shape, "standard": std}
    return distribution

def l1(model, rdo=1e2):
    with torch.no_grad():
        N = 0
        for param in model.parameters():
            N += param.numel()
        for param in model.parameters():
            param.data -= rdo / N * torch.sign(param.data)