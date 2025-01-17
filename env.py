import torch

def get_env():
    env = torch.zeros((1,1,32,32), dtype=torch.float32)
    env[0,0,4:-4, 4:-4]=1
    return env