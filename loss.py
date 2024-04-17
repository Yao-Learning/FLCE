import torch
import torch.nn as nn


class SupConLoss2(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, temperature:float=0.07, contrast_mode:str="all", base_temperature:float=0.07, device=None):
        super(SupConLoss2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.__device = torch.device('cuda:0')

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model.
        If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        check = lambda x: not bool(torch.any(torch.isnan(x)))
        

        if len(features.shape) < 3:
            raise ValueError("[features] needs to be [bsz, n_views, ...], at least 3 dimensions are required")
        elif len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both [labels] and [mask]")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            # mask is None
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float()
        else:
            # mask is not None & labels is None
            assert isinstance(mask, torch.Tensor)
            mask = mask.float()  # shape [bsz, bsz]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # shape [bsz, ...]
        print("contrast_feature", check(contrast_feature))
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode [{self.contrast_mode}]")

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )  # shape [bsz, bsz]
        
        print("anchor_dot_contrast", check(anchor_dot_contrast))
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # shape [bsz, 1]
        logits = anchor_dot_contrast - logits_max.detach()  # shape [bsz, bsz]

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # shape [bsz, bsz]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask).to('cpu'),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to('cpu'),
            0
        )  # shape [bsz, bsz]
        logits_mask = logits_mask.to('cuda:0')
        mask = mask.to('cuda:0')
        mask = mask * logits_mask
        
        print("logits", check(logits))
        print("logits_mask", check(logits_mask))

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # shape [bsz, bsz]
        
        print("exp_logits", check(exp_logits))
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # shape [bsz, bsz]
        
        print("log_prob", check(log_prob))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # shape [bsz]
        
        print("mean_log_prob_pos", check(mean_log_prob_pos))

        # loss
        loss_1 = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # shape [bsz]

        # BUG
        loss_2 = loss_1.view(anchor_count, batch_size)  # shape [1, bsz]
        loss = loss_2.mean()

        return loss