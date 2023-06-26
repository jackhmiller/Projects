from torch import nn
import torch
import torch.nn.functional as F


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()

	def forward(self, input, target):
		smooth = 1.0
		input_flat = input.view(-1)
		target_flat = target.view(-1)

		intersection = (input_flat * target_flat).sum()

		dice_score = (2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
		dice_loss = 1.0 - dice_score

		return dice_loss


class FocalLoss(nn.Module):

	def __init__(self, gamma=2, alpha=None):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, input, target):
		ce_loss = F.cross_entropy(input, target, reduction='none')

		pt = torch.exp(-ce_loss)
		focal_loss = (1 - pt) ** self.gamma * ce_loss

		if self.alpha is not None:
			alpha_weights = torch.full_like(target, self.alpha)
			alpha_weights[target == 0] = 1 - self.alpha
			focal_loss = focal_loss * alpha_weights

		return focal_loss.mean()
