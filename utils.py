import torch

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_no_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



if __name__ == "__main__":
    pass

