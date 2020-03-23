import numpy as np
import torch
import torch.nn.functional as F



class MultiBoxLoss(torch.nn.Module):

    def __init__(self, negative_ratio=3.0):
        super(MultiBoxLoss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred_conf, pred_loc, gt_conf, gt_loc):
        """ softmax cross entropy """
        pred_conf_log_softmax = F.log_softmax(pred_conf, -1)
        conf_loss = -torch.sum(pred_conf_log_softmax * gt_conf, -1)
        """ positive mask """
        positive_conf_mask_np = gt_conf.detach().cpu().numpy()[:, :, 0] == 0
        positive_conf_mask_np = positive_conf_mask_np.astype(np.float32)
        num_positive_conf = np.count_nonzero(positive_conf_mask_np)
        positive_conf_mask = torch.from_numpy(positive_conf_mask_np).to(pred_conf.device)
        """ positive conf loss """
        positive_conf_loss = positive_conf_mask * conf_loss
        """ negative conf mask """
        negative_conf_mask_np = 1 - positive_conf_mask_np
        conf_loss_np = conf_loss.detach().cpu().numpy() * negative_conf_mask_np
        conf_loss_argsort = np.argsort(conf_loss_np)
        num_negative_conf = int(self.negative_ratio * num_positive_conf)
        conf_loss_threshold = conf_loss_argsort[:, num_negative_conf]
        conf_loss_threshold = np.expand_dims(conf_loss_threshold, axis=-1)
        conf_loss_threshold = np.tile(conf_loss_threshold, [1, conf_loss.shape[-1]])
        threshed_conf_mask_np = conf_loss_np > conf_loss_threshold
        negative_conf_mask_np *= threshed_conf_mask_np.astype(np.float32)
        negative_conf_mask = torch.from_numpy(negative_conf_mask_np).to(pred_conf.device)
        """ negative conf loss """
        negative_conf_loss = negative_conf_mask * conf_loss
        """ conf loss """
        conf_loss = positive_conf_loss + negative_conf_loss
        conf_loss = torch.sum(conf_loss, -1)
        """ loc """
        loc_diff = pred_loc - gt_loc
        loc_linear_mask_np = np.abs(loc_diff.detach().cpu().numpy()) > 1.0
        loc_linear_mask_np = loc_linear_mask_np.astype(np.float32)
        loc_linear_mask = torch.from_numpy(loc_linear_mask_np).to(pred_loc.device)
        loc_loss_linear = (torch.abs(loc_diff) - 0.5) * loc_linear_mask
        loc_loss_square = 0.5 * loc_diff * loc_diff * (1 - loc_linear_mask)
        loc_loss_mask_np = gt_loc.detach().cpu().numpy() > 1e-9
        loc_loss_mask_np = loc_linear_mask_np.astype(np.float32)
        loc_loss_mask = torch.from_numpy(loc_loss_mask_np).to(pred_loc.device)
        loc_loss = (loc_loss_square + loc_loss_linear) * loc_loss_mask
        loc_loss = torch.sum(torch.sum(loc_loss, -1), -1)
        return conf_loss, loc_loss