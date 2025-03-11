import torch
import torch.nn as nn

cross_entropy = nn.CrossEntropyLoss(reduction='mean')

def compute_loss(scores, target):
    return cross_entropy(scores, target)

def compute_similarity(q_reps, p_reps):
    if not isinstance(q_reps, torch.Tensor):
        q_reps = torch.tensor(q_reps)
    if not isinstance(p_reps, torch.Tensor):
        p_reps = torch.tensor(p_reps)
    return torch.matmul(q_reps, p_reps.transpose(0,1))