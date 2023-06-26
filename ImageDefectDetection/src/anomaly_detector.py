import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import numpy as np
from PIL import Image
from numpy import uint8
from numpy.typing import NDArray

from deep_model.model import UNet


def classic_detector(image: NDArray[uint8], template: NDArray[uint8], case: str, threshold: int = 80):
	"""
	Pixel-wise Comparison between the image and the template by calculating the absolute difference
	between corresponding pixels of the image and template (based on a predefined threshold)
	"""
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	diff = cv2.absdiff(gray_image, gray_template)

	_, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	segmented_image = cv2.bitwise_and(image, image, mask=mask)

	print('Saving binary mask image')
	cv2.imwrite(f"results/{case}_classic_final.png", segmented_image)

	return


def deep_detector(image: NDArray[uint8], case: str):

	original_height, original_width = tuple(image.shape[:2])

	inference_transform = transforms.Compose([
		transforms.Resize((256, 256), interpolation=Image.BILINEAR),
		transforms.ToTensor(),
	])

	image_trans = inference_transform(Image.fromarray(image))
	image_trans = image_trans.unsqueeze(0)

	output_img, img_mask = predict(image_trans)

	pil_mask = TF.to_pil_image(img_mask)
	out = TF.to_pil_image(output_img[0])
	resized_mask = cv2.resize(np.array(pil_mask), (original_width, original_height))
	resized_output = cv2.resize(np.array(out), (original_width, original_height))

	print('Saving binary mask image')
	cv2.imwrite(f"results/{case}_deep_out_final.png", resized_output)
	cv2.imwrite(f"results/{case}_deep_mask_final.png", resized_mask)

	return


def predict(img, device='cpu'):
	model = UNet(in_channels=3, out_channels=1)
	model.load_state_dict(torch.load('deep_model/model_weights.pth'))
	model.eval()
	with torch.no_grad():
		image = img.to(device)
		output = model(image)
		predicted_mask = (output.squeeze() >= 0.5).float().cpu().numpy()

	return output, predicted_mask