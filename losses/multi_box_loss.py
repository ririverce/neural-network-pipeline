import numpy as np
import torch
import torch.nn.functional as F



class MultiBoxLoss(torch.nn.Module):

    def __init__(self, negative_ratio=3.0):
        super(MultiBoxLoss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred_conf, pred_loc, gt_conf, gt_loc):
        """******************
        ***** conf loss *****
        ******************"""
        """ Cross Entropy Loss """
        conf_loss = -1 * F.log_softmax(pred_conf, -1) * gt_conf
        conf_loss = conf_loss.sum(-1)
        """ Positive Loss """
        positive_conf_mask_np = gt_conf.detach().cpu().numpy()[:, :, 0] == 0
        positive_conf_mask = torch.from_numpy(positive_conf_mask_np).to(pred_conf.device)
        num_positive_conf = positive_conf_mask.sum(-1)
        positive_conf_loss = conf_loss * positive_conf_mask
        positive_conf_loss = positive_conf_loss.sum(-1)
        """ Hard Negative Mining """
        num_negative_conf = num_positive_conf * self.negative_ratio
        num_negative_conf_np = num_positive_conf.detach().cpu().numpy()
        negative_conf_loss = conf_loss * positive_conf_mask.logical_not()
        negative_conf_loss_np = negative_conf_loss.detach().cpu().numpy()
        negative_conf_mask_np = np.stack(
                                    [loss > loss[np.argsort(loss)[::-1][num]]] \
                                    for loss, num in zip(negative_conf_loss_np,
                                                         num_negative_conf_np)
                                ).reshape([pred_conf.shape[0], -1])
        negative_conf_mask = torch.from_numpy(negative_conf_mask_np)
        negative_conf_mask = negative_conf_mask.to(conf_loss.device)
        negative_conf_loss = conf_loss * negative_conf_mask
        negative_conf_loss = negative_conf_loss.sum(-1)
        """ total """
        positive_conf_loss = positive_conf_loss / num_positive_conf
        negative_conf_loss = negative_conf_loss / num_negative_conf
        conf_loss = positive_conf_loss + negative_conf_loss
        """*****************
        ***** loc loss *****
        *****************"""
        """ Positive Mask """
        loc_loss_mask_np = gt_loc.detach().cpu().numpy() != 0
        loc_loss_mask = torch.from_numpy(loc_loss_mask_np).to(pred_loc.device)
        num_positive_loc = loc_loss_mask.max(-1)[0].sum(-1)
        """ Smooth L1 Loss """
        loc_diff = torch.abs(pred_loc - gt_loc)
        loc_diff_mask_np = loc_diff.detach().cpu().numpy() < 1.0
        loc_diff_mask = torch.from_numpy(loc_diff_mask_np).to(loc_diff.device)
        loc_loss_linear = (loc_diff - 0.5) * loc_diff_mask.logical_not()
        loc_loss_square = 0.5 * loc_diff.pow(2) * loc_diff_mask
        loc_loss = (loc_loss_linear + loc_loss_square) * loc_loss_mask
        loc_loss = loc_loss.sum(-1).sum(-1)
        loc_loss = loc_loss / num_positive_loc
        return conf_loss, loc_loss