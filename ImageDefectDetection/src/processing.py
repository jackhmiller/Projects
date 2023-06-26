from typing import Tuple, Any

import cv2
import numpy as np
from numpy import uint8, ndarray, dtype
from numpy.typing import NDArray

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.10


def run_image_processing(ref_path: str, inspect_path: str, case: str) -> Tuple[Any, ndarray]:
	print("Reading reference image: ", ref_path)
	ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)

	print("Reading image to align: ", inspect_path)
	inspect_img = cv2.imread(inspect_path, cv2.IMREAD_COLOR)

	preprocessor = ImageProcessor(inspect_img, ref_img, case)
	preprocessor.align_images()
	cropped_inspect, cropped_ref = preprocessor.crop_after_alignment()

	print("Completed image processing")

	return cropped_inspect, cropped_ref


class ImageProcessor:
	def __init__(self, img_inspect: NDArray[uint8], img_reference: NDArray[uint8], case: str):
		self.img_inspect = img_inspect
		self.gray_inspect = cv2.cvtColor(img_inspect, cv2.COLOR_BGR2GRAY)
		self.img_reference = img_reference
		self.gray_ref = cv2.cvtColor(self.img_reference, cv2.COLOR_BGR2GRAY)
		self.case = case
		self.aligned_inspect = None

	def resize_images(self):
		"""
		Resize original images if necessary prior to feature matching and perspective transform.
		"""
		print('Resizing images')
		h1, w1, _ = self.img_inspect.shape
		h2, w2, _ = self.img_reference.shape

		max_width = max(w1, w2)
		max_height = max(h1, h2)

		self.img_inspect = cv2.resize(self.img_inspect, (max_width, max_height))
		self.img_reference = cv2.resize(self.img_reference, (max_width, max_height))

		return

	def align_images(self):
		"""
		Use homography matrix to perform the perspective transformation.
		"""
		if self.img_reference.shape != self.img_inspect.shape:
			self.resize_images()

		homography = self.feature_match()

		height, width, _ = self.img_reference.shape

		self.aligned_inspect = cv2.warpPerspective(self.img_inspect, homography, (width, height))

		return

	def feature_match(self):
		"""
		Use ORB algorithm to detect ORB features based on the intensity values and local binary patterns
		within the image patches. Use brute-force matching with Hamming distance to match descriptors, sort them by
		distance, and threshold top matches to be used in final homography calculation.
		"""
		orb = cv2.ORB_create(MAX_FEATURES)
		keypoints1, descriptors1 = orb.detectAndCompute(self.gray_inspect, None)
		keypoints2, descriptors2 = orb.detectAndCompute(self.gray_ref, None)

		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		matches = bf.match(descriptors1,
						   descriptors2)

		matches = sorted(matches, key=lambda x: x.distance)
		n_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
		matches = matches[:n_good_matches]

		img_features = cv2.drawMatches(self.gray_inspect, keypoints1, self.gray_ref, keypoints2, matches, None)
		cv2.imwrite(f"results/{self.case}_feature_matches.jpg", img_features)

		points1 = np.zeros((len(matches), 2), dtype=np.float32)
		points2 = np.zeros((len(matches), 2), dtype=np.float32)

		for i, match in enumerate(matches):
			points1[i, :] = keypoints1[match.queryIdx].pt
			points2[i, :] = keypoints2[match.trainIdx].pt

		homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

		return homography

	def crop_after_alignment(self):
		"""
		Threshold the grayscale image to create a binary mask, then find non-zero contours of the mask. Use those
		contours to create bboxes to subsequently be cropped.
		"""
		aligned_gray = cv2.cvtColor(self.aligned_inspect, cv2.COLOR_BGR2GRAY)

		_, thresh = cv2.threshold(aligned_gray, 1, 255, cv2.THRESH_BINARY)

		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		x, y, w, h = cv2.boundingRect(np.concatenate(contours))

		cropped_inspect = self.aligned_inspect[y:y + h, x:x + w]
		cropped_ref = self.img_reference[y:y + h, x:x + w]

		assert cropped_inspect.shape == cropped_ref.shape

		return cropped_inspect, cropped_ref
