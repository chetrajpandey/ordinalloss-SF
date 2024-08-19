import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, alpha=1, reduction='mean'):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, log_scale):
        # Move weight tensor to the same device as inputs
        if self.weight is not None:
            self.weight = self.weight.to(inputs.device)

        # Move inputs, targets, and log_scale to the device
        inputs = inputs.to(inputs.device)
        targets = targets.to(inputs.device)
        log_scale = log_scale.to(inputs.device)

        # Calculate cross-entropy term
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')

        # Calculate difficulty term
        difficulty_term  = 1/(torch.log10(log_scale))  # Avoid division by zero

        # Calculate final focal loss
        loss = self.alpha * ce_loss * difficulty_term

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")
