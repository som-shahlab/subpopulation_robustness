import numpy as np
import torch
import torch.nn.functional as F

# Metric functions that have the same function call signature as torch loss functions
# Details
# Operate on unnormalized model outputs
# Take a sample_weight argument
# Take a surrogate_fn argument for smooth differentiable relaxations of the metric


def roc_auc_score_surrogate(outputs, labels, sample_weight=None, surrogate_fn=None):
    """
        The area under the ROC score
    """

    pos_mask = labels == 1
    neg_mask = labels == 0

    if (pos_mask.sum() == 0) or (neg_mask.sum() == 0):
        raise MetricUndefinedError

    if surrogate_fn is None:
        surrogate_fn = logistic_surrogate

    outputs = F.log_softmax(outputs, dim=1)[:, -1]

    preds_pos = outputs[pos_mask]
    preds_neg = outputs[neg_mask]

    preds_difference = preds_pos.unsqueeze(0) - preds_neg.unsqueeze(1)

    if sample_weight is None:
        result = surrogate_fn(preds_difference).mean()
        return result
    else:
        weights_pos = sample_weight[pos_mask]
        weights_neg = sample_weight[neg_mask]
        weights_tile = weights_pos.unsqueeze(0) * weights_neg.unsqueeze(1)
        return (
            surrogate_fn(preds_difference) * weights_tile
        ).sum() / weights_tile.sum()


def sigmoid(x, surrogate_scale=1.0):
    return torch.sigmoid(x * surrogate_scale)


def logistic_surrogate(x):
    # See Bishop PRML equation 7.48
    return torch.nn.functional.softplus(x) / torch.tensor(np.log(2, dtype=np.float32))


def hinge_surrogate(x):
    return torch.nn.functional.relu(1 + x)


def indicator(x):
    return 1.0 * (x > 0)


class MetricUndefinedError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def weighted_mean(x, sample_weight=None):
    """
    A simple torch weighted mean function
    """
    if sample_weight is None:
        return x.mean()
    else:
        assert x.shape == sample_weight.shape
        return (x * sample_weight).sum() / sample_weight.sum()


def weighted_cross_entropy_loss(outputs, labels, sample_weight=None, **kwargs):
    """
    A method that computes a sample weighted cross entropy loss
    """
    if sample_weight is None:
        return F.cross_entropy(outputs, labels, reduction="mean")
    else:
        result = F.cross_entropy(outputs, labels, reduction="none", **kwargs)
        assert result.size()[0] == sample_weight.size()[0]
        return (sample_weight * result).sum() / sample_weight.sum()


def baselined_loss(outputs, labels, sample_weight, **kwargs):
    return weighted_cross_entropy_loss(
        outputs, labels, sample_weight=sample_weight, **kwargs
    ) - bernoulli_entropy(labels, sample_weight=sample_weight)


def bernoulli_entropy(x, sample_weight=None, eps=1e-6):
    """
        Computes Bernoulli entropy
    """

    if sample_weight is None:
        x = x.float().mean()
    else:
        x = (sample_weight * x).sum() / sample_weight.sum()

    if ((1 - x) < eps) or (x < eps):
        return torch.FloatTensor([0]).to(x.device)

    return -((torch.log(x) * x)) + (torch.log(1 - x) * (1 - x))
