import ignite.contrib.metrics as icm
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score
from torch import Tensor
import numpy as np

# functions

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)



def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def binarize(input):
    '''
    :param input:
    :return: binarized input
    '''
    return input >= 0.5


# Loss
class BCELoss(nn.Module): ## Problematic

    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, output, target):
        print(output.reshape(target.shape), target)

        return self.loss(output.reshape(target.shape), target)


class BCEWithLogitsLoss(nn.Module):

    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, output, target):
        return self.loss(output, target)



class WeightedBCELoss2(nn.Module):
    def __init__(self, class_weights=None):
        super(WeightedBCELoss2, self).__init__()
        self.register_buffer('class_weights', class_weights)

    def forward(self, output, target):
        weight=self.class_weights
        if weight is not None:
            assert len(weight) == 2
            weight_ = weight[target.long()]
            loss = nn.BCELoss(weight=weight_)
        else:
            loss = nn.BCELoss()

        return loss(output.reshape(target.shape), target)



# Synthesis
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, output, target):
        return self.loss(output, target)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss(), self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        return self.loss(output, target)



# Evaluation loss

class AUC:
    """Computes AUC.
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, output, target):
        output, target = convert_to_numpy(output, target)
        if len(np.unique(target)) != 2:
            return 0

        return roc_auc_score(target, output)

class F1:
    """Computes AUC.
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, output, target):
        output, target = convert_to_numpy(output, target)
        return f1_score(target, binarize(output))
