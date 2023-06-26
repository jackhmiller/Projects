import cv2
import os
from anomaly_detector import classic_detector, deep_detector
from processing import run_image_processing
from utilities import ssim

SSIM_THRESH = 0.8


def get_case_name(filename: str) -> str:
	return filename.split('/')[1].split('_')[0]


def main(reference_img_path: str, inspect_img_path: str, method: str):
	"""
	Align the images, compute SSIM to determine if anomaly present or not, then run the anoamly detector.
	"""
	case_name = get_case_name(inspect_img_path)

	inspect_processed, ref_processed = run_image_processing(ref_path=reference_img_path, inspect_path=inspect_img_path, case=case_name)

	ssim_score = ssim(img1=inspect_processed, img2=ref_processed)

	if ssim_score < SSIM_THRESH:
		print(f"Defect found in {case_name}!")

		if method == 'classic':
			classic_detector(inspect_processed, ref_processed, case_name)
		else:
			deep_detector(inspect_processed, case_name)
	else:
		print(f"No defect for {case_name}")


if __name__ == '__main__':
	images_path = "src_images"
	reference_filename = "defective_examples/case1_reference_image.tif"
	inspect_filename = "defective_examples/case1_inspected_image.tif"
	method = 'deep'  # or 'classic

	main(reference_img_path=os.path.join(images_path, reference_filename),
		 inspect_img_path=os.path.join(images_path, inspect_filename),
		 method=method)
