import mmcv
import os.path as osp
from mmcv import Config
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,train_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
import time
import cv2
import os
import json
from collections import OrderedDict
import numpy as np
from mmpose.core.evaluation.top_down_eval import (keypoint_nme,
                                                  keypoint_pck_accuracy)
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt2dSviewRgbImgTopDownDataset


class MMPose:
    def __init__(self, 
        backbone_det='FasterRCNN-pose',
        backbone = 'HrNet32', 
        #note if inference need HrNet (64)
        dataset_path = None
        ):

        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))

        self.backbone_det = backbone_det
        backbone_det_path = os.path.join(self.file_dirname, 'models', self.backbone_det)
        ckpt_cfg_list = list(os.listdir(backbone_det_path))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.det_config = os.path.join(backbone_det_path, item)
            elif item[-1] == 'h':
                self.det_checkpoint = os.path.join(backbone_det_path, item)
            else:
                print("Warning!!! There is an unrecognized file in the backbone folder.")

        self.backbone = backbone
        backbone_path = os.path.join(self.file_dirname, 'models', self.backbone)
        ckpt_cfg_list = list(os.listdir(backbone_path))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.pose_config = os.path.join(backbone_path, item)
            elif item[-1] == 'h':
                self.pose_checkpoint = os.path.join(backbone_path, item)
            else:
                print("Warning!!! There is an unrecognized file in the backbone folder.")

        self.cfg_det = Config.fromfile(self.det_config)
        self.cfg = Config.fromfile(self.pose_config)
        
        self.dataset_path = dataset_path

        return None


    def train(self, random_seed=0, save_fold=None, checkpoint = None, distributed=False, validate=True,
              metric='PCK', save_best = 'PCK',optimizer="Adam", epochs=100, lr=5e-4):

        # 如果外部不指定save_fold
        if not self.save_fold:
            # 如果外部也没有传入save_fold，我们使用默认路径
            if not save_fold:
                self.save_fold = os.path.join(self.cwd, 'checkpoints/pose_model')
            # 如果外部传入save_fold，我们使用传入值
            else:
                self.save_fold = save_fold

        # self.cfg = Config.fromfile(self.backbonedict[self.backbone])
        # print(self.cfg.pretty_text)
        self.cfg.gpu_ids = range(1)
        self.cfg.work_dir = self.save_fold
        self.cfg.load_from = checkpoint
        self.cfg.seed = random_seed
        # self.cfg.model.backbone.frozen_stages = Frozen_stages
        # set log interval
        self.cfg.log_config.interval = 1
        self.cfg.total_epochs = epochs  # 最大的训练轮次
        self.cfg.optimizer.lr = lr  # 学习率
        self.cfg.optimizer.type = optimizer  # 优化器
        self.cfg.evaluation.metric = metric  # 验证指标
        self.cfg.evaluation.save_best = save_best  # 验证指标
    
        datasets = [build_dataset(self.cfg.data.train)]

        # build model
        model = build_posenet(self.cfg.model)

        # create work_dir
        mmcv.mkdir_or_exist(self.cfg.work_dir)

        # train model
        train_model(
            model, datasets, self.cfg, distributed=distributed, validate=validate, meta=dict())

        return None

        
    def inference(self,
                  device='cuda:0',
                  is_trained=False,
                  pretrain_model='./checkpoints/pose_model/latest.pth',
                  img=None,
                  show=True,
                  save=True,
                  work_dir='./',
                  name='pose_result'):
        """
        params:
            device: 推理设备,可选参数: ('cuda:int','cpu')
            is_trained: 是否使用本地预训练的其他模型进行训练
            pretrain_model: 如果使用其他模型，则传入模型路径
            img: 推理图片的路径
            show: 是否对推理结果进行显示
            save: 是否对推理结果进行保存
            work_dir: 推理结果图片的保存文件夹
            name：推理结果保存的名字
        return:
            pose_results: 推理的结果数据，一个列表，其中包含若干个字典，每个字典存储对应检测的人体数据。
        """

        if not pretrain_model:
            pretrain_model = os.path.join(self.cwd, 'checkpoints/pose_model/latest.pth')
        print("========= begin inference ==========")

        if is_trained == True:
            self.pose_checkpoint = pretrain_model

        # initialize pose model
        pose_model = init_pose_model(self.pose_config, self.pose_checkpoint,device = device)
        # initialize detector
        det_model = init_detector(self.det_config, self.det_checkpoint,device=device)

        # inference detection
        mmdet_results = inference_detector(det_model, img)

        # extract person (COCO_ID=1) bounding boxes from the detection results
        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        # inference pose
        pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                                    img,
                                                                    person_results,
                                                                    bbox_thr=0.3,
                                                                    format='xyxy',
                                                                    dataset=pose_model.cfg.data.test.type)

        # show pose estimation results
        vis_result = vis_pose_result(pose_model,
                                    img,
                                    pose_results,
                                    dataset=pose_model.cfg.data.test.type,
                                    show=show)

        # vis_result = cv2.resize(vis_result, dsize=None, fx=1, fy=1)

        # 如果不保存则直接返回推理结果
        if not save:
            return pose_results
        from IPython.display import Image, display
        import tempfile
        import os.path as osp
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = osp.join(work_dir, name+'.png')
            cv2.imwrite(file_name, vis_result)
            display(Image(file_name))
        return pose_results

    def load_dataset(self, path):

        self.dataset_path = path

        #数据集修正为 images train.json val.json 形式
        # cfg.data_root = 'data/coco_tiny'
        self.cfg.data.train.type = 'PoseDataset'
        
        self.cfg.data.train.ann_file = os.path.join(self.dataset_path, 'train.json')
        self.cfg.data.train.img_prefix = os.path.join(self.dataset_path, 'images/')

        self.cfg.data.val.type = 'PoseDataset'
        self.cfg.data.val.ann_file = os.path.join(self.dataset_path, 'val.json')
        self.cfg.data.val.img_prefix = os.path.join(self.dataset_path, 'images/')

        self.cfg.data.test.type = 'PoseDataset'
        self.cfg.data.test.ann_file = os.path.join(self.dataset_path, 'val.json')
        self.cfg.data.test.img_prefix = os.path.join(self.dataset_path, 'images/')


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
