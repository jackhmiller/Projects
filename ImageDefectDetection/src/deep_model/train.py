import torch
from model import UNet
from dataloader import get_dataloaders, get_datasets
from loss_functions import DiceLoss

PATH = r'/submission/src_images/mask'


def train_model():
	train_dataset, val_dataset = get_datasets(PATH)
	loaders = get_dataloaders(train_dataset, val_dataset)

	device = torch.device("cpu")
	dsc_loss = DiceLoss()
	best_validation_dsc = 1.0

	lr = 0.001
	epochs = 200

	unet = UNet(in_channels=3, out_channels=1)
	unet.to(device)
	optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

	best_val_true = []
	best_val_pred = []
	epoch_loss = {"train": [], "valid": []}

	for epoch in range(epochs):
		print('-' * 100)

		for phase in ["train", "valid"]:

			if phase == "train":
				unet.train()
			else:
				unet.eval()

			epoch_samples = 0
			running_loss = 0

			for i, data in enumerate(loaders[phase]):
				x, y_true = data
				x, y_true = x.to(device), y_true.to(device)

				epoch_samples += x.size(0)

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == "train"):
					y_pred = unet(x)
					loss = dsc_loss(y_pred, y_true)
					running_loss += loss.item()

					if phase == "train":
						loss.backward()
						optimizer.step()

			epoch_phase_loss = running_loss / epoch_samples
			epoch_loss[phase].append(epoch_phase_loss)

			if phase == "valid" and epoch_phase_loss < best_validation_dsc:
				best_validation_dsc = epoch_phase_loss
				print("Saving best model")
				torch.save(unet.state_dict(), 'model_weights.pth')
				best_val_true = y_true.detach().cpu().numpy()
				best_val_pred = y_pred.detach().cpu().numpy()

		print(f"Epoch {epoch + 1}/{epochs}: Train loss = {epoch_loss['train'][epoch]} --- Validation loss = {epoch_loss['valid'][epoch]}")

	print("Training completed.")


if __name__ == '__main__':
	train_model()