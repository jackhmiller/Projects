from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import cv2
import random

IMG_DEST = r'\src_images\images' 
MASK_DEST = r'\src_images\mask'


class AnomalyDataset(Dataset):
	def __init__(self, images_directory, masks_directory, mask_filenames, transform):
		self.images_directory = images_directory
		self.masks_directory = masks_directory
		self.mask_filenames = mask_filenames
		self.transform = transform

	def __len__(self):
		return len(self.mask_filenames)

	def __getitem__(self, idx):
		mask_filename = self.mask_filenames[idx]
		image = cv2.imread(os.path.join(self.images_directory, mask_filename))
		mask = cv2.imread(os.path.join(self.masks_directory, mask_filename),
						  cv2.IMREAD_UNCHANGED)

		image = self.transform(Image.fromarray(image))
		mask = self.transform(Image.fromarray(mask))

		return image, mask


def get_datasets(img_dir: str, n_val: int = 10):
	mask_filenames = list(sorted(os.listdir(img_dir)))
	random.shuffle(mask_filenames)

	val_mask_filenames = mask_filenames[:n_val]
	train_mask_filenames = mask_filenames[n_val:]

	train_transform = transforms.Compose([
		transforms.Resize((256, 256), interpolation=Image.BILINEAR),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomRotation(degrees=30),
		transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
		transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
		transforms.ToTensor(),
	])

	val_transform = transforms.Compose([
		transforms.Resize((256, 256), interpolation=Image.BILINEAR),
		transforms.ToTensor(),
	])

	train_dataset = AnomalyDataset(IMG_DEST, MASK_DEST, train_mask_filenames, transform=train_transform)

	val_dataset = AnomalyDataset(IMG_DEST, MASK_DEST, val_mask_filenames, transform=val_transform)

	return train_dataset, val_dataset


def get_dataloaders(dataset_train, dataset_valid, batch_size=5, num_workers=0):

	loader_train = DataLoader(
		dataset_train,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=num_workers,
	)

	loader_valid = DataLoader(
		dataset_valid,
		batch_size=batch_size,
		drop_last=False,
		shuffle=False,
		num_workers=num_workers,
	)

	return {"train": loader_train, "valid": loader_valid}
