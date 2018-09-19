"""
Loss functions used for recommender models.
"""

import torch


def sigmoid_log_loss(positive_predictions, negative_predictions):
    """
    The point-wise log loss (binary cross-entropy) function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    """
    loss1 = -torch.log(torch.sigmoid(positive_predictions))
    loss0 = -torch.log(1 - torch.sigmoid(negative_predictions))

    # loss = torch.cat((loss1.view(-1), loss0.view(-1))).mean()
    loss = torch.sum(torch.cat((loss1, loss0), 1), dim=1)

    return loss.mean()


def weighted_sigmoid_log_loss(positive_predictions, negative_predictions, candidate_predictions, weight, alpha=1.0):
    """
    The weighted point-wise log loss (binary cross-entropy) function used for ranking distillation.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    candidate_predictions:
        Tensor containing predictions for teacher's top-k ranked items.
    weight:
        Tensor containing weight for every loss term.
    alpha: float, optional
        Weight for balancing ranking loss and distillation loss.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    reg_loss, float
        The mean value of regular point-wise loss function (i.e., without distillation loss).
    """
    loss1 = -torch.log(torch.sigmoid(positive_predictions))
    loss0 = -torch.log(1 - torch.sigmoid(negative_predictions))

    loss_cand = -torch.log(torch.sigmoid(candidate_predictions))

    if weight is not None:
        loss_cand = loss_cand * weight.expand_as(loss_cand)

    if alpha is not None:
        loss_cand = loss_cand * alpha

    loss = torch.sum(torch.cat((loss1, loss0, loss_cand), 1), dim=1)
    reg_loss = torch.sum(torch.cat((loss1, loss0), 1), dim=1)

    return loss.mean(), reg_loss.mean()
