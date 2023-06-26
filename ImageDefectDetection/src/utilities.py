import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from numpy import uint8
from numpy.typing import NDArray
import hashlib
from datetime import datetime
import os
import shutil

img_folder = r'\Downloads\grid\test'  # obviously bad practice to put paths like this, but for time's sake...
mask_folder = r'\Downloads\grid\ground_truth'
img_dest = r'\muze\submission\src_images\images'
mask_dest = r'C:\muze\submission\src_images\mask'


def create_train_label_folders() -> None:
	folders = [(img_folder, img_dest), (mask_folder, mask_dest)]
	for f in folders:
		if not os.path.exists(f[1]):
			os.makedirs(f[1])

		subfolders = [f.path for f in os.scandir(f[0]) if f.is_dir()]
		idx = 1
		for source_dir in subfolders:
			file_list = os.listdir(source_dir)

			for file_name in file_list:
				source_file = os.path.join(source_dir, file_name)
				destination_file = os.path.join(f[1], f"img_{idx}.png")
				shutil.copyfile(source_file, destination_file)
				idx += 1


def generate_run_id():
	return hashlib.sha1(datetime.now().strftime("%Y-%m-%d, %H:%M:%S").encode()).hexdigest()


def _gaussian(window_size: int, sigma: float):
	"""
	Create Gaussian noise.
	"""
	gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
	return gauss / gauss.sum()


def _create_window(window_size: int, channel: int):
	_1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

	return window


def calc_ssim(img1, img2, window, window_size: int, channel: int, size_average=True):
	"""
	Compare images patch-wise using summary statistics (mean, var, covar) to calculate scale-invariant
	luminance score (0 means a large difference), contrast score, and structure store. The final SSIM metric
	is a product of these three scores.
	"""
	mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
	mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

	# division-stabilizing constants
	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1: NDArray[uint8], img2: NDArray[uint8], window_size=11, size_average=True):
	img1_tensor = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0
	img2_tensor = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0

	(_, channel, _, _) = img1_tensor.size()
	window = _create_window(window_size, channel)

	window = window.type_as(img1_tensor)

	score = calc_ssim(img1_tensor, img2_tensor, window, window_size, channel, size_average)

	return score.detach().numpy()