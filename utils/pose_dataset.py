import json
import os
import os.path as osp
from collections import OrderedDict

import numpy as np

from mmpose.core.evaluation.top_down_eval import (keypoint_nme,
                                                  keypoint_pck_accuracy)
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class PoseDataset(Kpt2dSviewRgbImgTopDownDataset):

	def __init__(self,
				 ann_file,
				 img_prefix,
				 data_cfg,
				 pipeline,
				 dataset_info=None,
				 test_mode=False):
		super().__init__(
			ann_file, img_prefix, data_cfg, pipeline, dataset_info, coco_style=False, test_mode=test_mode)

		# flip_pairs, upper_body_ids and lower_body_ids will be used
		# in some data augmentations like random flip
		self.ann_info['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
									   [11, 12], [13, 14], [15, 16]]
		self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
		self.ann_info['lower_body_ids'] = (11, 12, 13, 14, 15, 16)

		self.ann_info['joint_weights'] = None
		self.ann_info['use_different_joint_weights'] = False

		self.dataset_name = 'coco_tiny'
		self.db = self._get_db()

	def _get_db(self):
		with open(self.ann_file) as f:
			anns = json.load(f)

		db = []
		for idx, ann in enumerate(anns):
			# get image path
			image_file = osp.join(self.img_prefix, ann['image_file'])
			# get bbox
			bbox = ann['bbox']
			center, scale = self._xywh2cs(*bbox)
			# get keypoints
			keypoints = np.array(
				ann['keypoints'], dtype=np.float32).reshape(-1, 3)
			num_joints = keypoints.shape[0]
			joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
			joints_3d[:, :2] = keypoints[:, :2]
			joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
			joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

			sample = {
				'image_file': image_file,
				'center': center,
				'scale': scale,
				'bbox': bbox,
				'rotation': 0,
				'joints_3d': joints_3d,
				'joints_3d_visible': joints_3d_visible,
				'bbox_score': 1,
				'bbox_id': idx,
			}
			db.append(sample)

		return db

	def _xywh2cs(self, x, y, w, h):
		"""This encodes bbox(x, y, w, h) into (center, scale)
		Args:
			x, y, w, h
		Returns:
			tuple: A tuple containing center and scale.
			- center (np.ndarray[float32](2,)): center of the bbox (x, y).
			- scale (np.ndarray[float32](2,)): scale of the bbox w & h.
		"""
		aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
			'image_size'][1]
		center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
		if w > aspect_ratio * h:
			h = w * 1.0 / aspect_ratio
		elif w < aspect_ratio * h:
			w = h * aspect_ratio

		# pixel std is 200.0
		scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
		# padding to include proper amount of context
		scale = scale * 1.25
		return center, scale

	def evaluate(self, outputs, res_folder, metric='PCK', **kwargs):
		"""Evaluate keypoint detection results. The pose prediction results will
		be saved in `${res_folder}/result_keypoints.json`.

		Note:
		batch_size: N
		num_keypoints: K
		heatmap height: H
		heatmap width: W

		Args:
		outputs (list(preds, boxes, image_path, output_heatmap))
			:preds (np.ndarray[N,K,3]): The first two dimensions are
				coordinates, score is the third dimension of the array.
			:boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
				, scale[1],area, score]
			:image_paths (list[str]): For example, ['Test/source/0.jpg']
			:output_heatmap (np.ndarray[N, K, H, W]): model outputs.

		res_folder (str): Path of directory to save the results.
		metric (str | list[str]): Metric to be performed.
			Options: 'PCK', 'NME'.

		Returns:
			dict: Evaluation results for evaluation metric.
		"""
		metrics = metric if isinstance(metric, list) else [metric]
		allowed_metrics = ['PCK', 'NME']
		for metric in metrics:
			if metric not in allowed_metrics:
				raise KeyError(f'metric {metric} is not supported')

		res_file = os.path.join(res_folder, 'result_keypoints.json')

		kpts = []
		for output in outputs:
			preds = output['preds']
			boxes = output['boxes']
			image_paths = output['image_paths']
			bbox_ids = output['bbox_ids']

			batch_size = len(image_paths)
			for i in range(batch_size):
				kpts.append({
					'keypoints': preds[i].tolist(),
					'center': boxes[i][0:2].tolist(),
					'scale': boxes[i][2:4].tolist(),
					'area': float(boxes[i][4]),
					'score': float(boxes[i][5]),
					'bbox_id': bbox_ids[i]
				})
		kpts = self._sort_and_unique_bboxes(kpts)

		self._write_keypoint_results(kpts, res_file)
		info_str = self._report_metric(res_file, metrics)
		name_value = OrderedDict(info_str)

		return name_value

	def _report_metric(self, res_file, metrics, pck_thr=0.3):
		"""Keypoint evaluation.

		Args:
		res_file (str): Json file stored prediction results.
		metrics (str | list[str]): Metric to be performed.
			Options: 'PCK', 'NME'.
		pck_thr (float): PCK threshold, default: 0.3.

		Returns:
		dict: Evaluation results for evaluation metric.
		"""
		info_str = []

		with open(res_file, 'r') as fin:
			preds = json.load(fin)
		assert len(preds) == len(self.db)

		outputs = []
		gts = []
		masks = []

		for pred, item in zip(preds, self.db):
			outputs.append(np.array(pred['keypoints'])[:, :-1])
			gts.append(np.array(item['joints_3d'])[:, :-1])
			masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)

		outputs = np.array(outputs)
		gts = np.array(gts)
		masks = np.array(masks)

		normalize_factor = self._get_normalize_factor(gts)

		if 'PCK' in metrics:
			_, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
											  normalize_factor)
			info_str.append(('PCK', pck))

		if 'NME' in metrics:
			info_str.append(
				('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

		return info_str

	@staticmethod
	def _write_keypoint_results(keypoints, res_file):
		"""Write results into a json file."""

		with open(res_file, 'w') as f:
			json.dump(keypoints, f, sort_keys=True, indent=4)

	@staticmethod
	def _sort_and_unique_bboxes(kpts, key='bbox_id'):
		"""sort kpts and remove the repeated ones."""
		kpts = sorted(kpts, key=lambda x: x[key])
		num = len(kpts)
		for i in range(num - 1, 0, -1):
			if kpts[i][key] == kpts[i - 1][key]:
				del kpts[i]

		return kpts
	
	@staticmethod
	def _get_normalize_factor(gts):
		"""Get inter-ocular distance as the normalize factor, measured as the
		Euclidean distance between the outer corners of the eyes.

		Args:
			gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

		Return:
			np.ndarray[N, 2]: normalized factor
		"""

		interocular = np.linalg.norm(
			gts[:, 0, :] - gts[:, 1, :], axis=1, keepdims=True)
		return np.tile(interocular, [1, 2])
