import torch
import torch.nn.functional as F



class CrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, gt):
        pred = F.log_softmax(pred, dim=-1)
        loss = -1 * torch.sum(pred * gt, dim=1)
        return loss