import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_rank


class SpearmanLoss(nn.Module):
    def __init__(self, reg_strength=1.0):
        super(SpearmanLoss, self).__init__()
        self.reg_strength = reg_strength

    def forward(self, pred, target):
        pred = soft_rank(pred, regularization_strength=self.reg_strength)
        target = soft_rank(target, regularization_strength=self.reg_strength)
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return 1 - (pred * target).sum()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CompositeLoss(nn.Module):
    def __init__(self, args, loss_func_wt, margin=1.0, temperature=1.0):
        super(CompositeLoss, self).__init__()
        self.args = args
        self.train_model = args.train_model
        self.mse_weight = float(loss_func_wt['MSE']) if 'MSE' in loss_func_wt.keys() else 0
        self.classification_weight = float(loss_func_wt['CLS']) if 'CLS' in loss_func_wt.keys() else 0

        if self.args.dataset_type == 'regression':
            self.sup_loss = nn.MSELoss(reduction='mean')
        elif self.args.dataset_type == 'classification':
            # self.sup_loss = nn.BCEWithLogitsLoss(reduction='mean', 
            #                                  pos_weight=torch.tensor([self.args.pos_weight]))
            self.sup_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
    def forward(self, output, query, support, reg_labels, cls_labels):
        output1, output2, output_reg, output_cls = output
        reg_label1, reg_label2, reg_label_res = reg_labels
        mol1, mol1_ = query[0], query[1]
        mol2, mol2_ = support[0], support[1]
        if self.args.dataset_type == 'regression':
            # MSE loss
            if self.mse_weight > 0:
                mse_loss = self.sup_loss(output1, reg_label1)
            else:
                mse_loss = torch.tensor(0).to(reg_label1.device)
            # classification loss
            if self.classification_weight > 0 and self.train_model not in ['KANO_Prot', 'KANO_ESM']:
                classification_loss = self.bce_loss(output_cls.squeeze(), cls_labels)
            else:
                classification_loss = torch.tensor(0).to(reg_label1.device)

            final_loss =  self.mse_weight * mse_loss +\
                        self.classification_weight * classification_loss

            return final_loss, mse_loss, torch.tensor(0).to(reg_label1.device), classification_loss

        elif self.args.dataset_type == 'classification':
            classification_loss = self.sup_loss(output1.squeeze(), reg_label1.squeeze())
            return classification_loss, None, None, None


class Ranked_SupCon_reg(nn.Module):
    def __init__(self, args, alpha=0.5, beta=0.5, 
                 threshold=8, temperature=0.07, reg_strength=1.0):
        super(Ranked_SupCon_reg, self).__init__()
        self.args = args
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.SupConLoss = SupConLoss(temperature=temperature)
        self.SpearmanLoss = SpearmanLoss(reg_strength=reg_strength)
        
    def forward(self, feat, pred, label):
        # idx=0 for anchor, idx=1 for positive, idx=2: for negative
        # feat: (N, V, D), N: batch size; V: number of view=2; D: embedding size
        # pred: (N, 1)
        # label: (N, 1)
        cls_label = torch.where(label > self.threshold, 
                        torch.tensor(1.0), torch.tensor(0.0)).to(pred.device)
        # true_ranks = torch.tensor(rankdata(label), dtype=torch.float32, device=pred.device) # true ranks: (N, M, 1)

        # calculate supervised loss
        if self.args.dataset_type == 'regression':
            sup_loss = self.mse_loss(pred, label)
        elif self.args.dataset_type == 'classification':
            sup_loss = self.bce_loss(pred, label)

        # calculate supervised contrastive loss
        if self.args.dataset_type == 'regression':
            supcon_loss = self.SupConLoss(feat, cls_label)
        elif self.args.dataset_type == 'classification':
            supcon_loss = self.SupConLoss(feat, label)

        # calculate Spearman ranking loss
        # representation-labels correlation
        # take one as an anchor, and calculate cosine similarity of features and absolute difference of predictions
        # cos_sim = F.cosine_similarity(feat[0, 0, :], feat[:, 0, :], dim=0)
        # pred_dist = torch.abs(pred - pred[0])
        # print(cos_sim.shape, pred_dist.shape)
        # r_p_corr_loss = self.SpearmanLoss(cos_sim, pred_dist)
        r_p_corr_loss = 0
        
        # predictions-labels correlation
        p_l_corr_loss = self.SpearmanLoss(pred, label)

        total_loss = sup_loss + self.alpha * supcon_loss + \
                                self.beta * (r_p_corr_loss + p_l_corr_loss)
        return total_loss, sup_loss, supcon_loss, [r_p_corr_loss, p_l_corr_loss]
