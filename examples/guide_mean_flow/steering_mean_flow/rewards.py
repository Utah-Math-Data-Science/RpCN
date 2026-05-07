import torch 

# circle rewards

def left(x):
    return -x[:, 0]

def bottom(x):
    return -x[:, 1]

def right(x):
    return x[:, 0]

def top(x):
    return x[:, 1]


# mixture rewards 

def left_node(x):

    target = torch.tensor([-4.0, 0.0], device=x.device)
    
    distance_sq = torch.sum((x - target)**2, dim=1)
    
    return -distance_sq
