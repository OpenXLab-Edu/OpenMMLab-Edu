import mmcv
import os.path as osp
from mmcv import Config
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,train_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from .pose_dataset import PoseDataset
import time
import cv2
import os



class MMPose:
    def __init__(self, 
        backbone_det='FasterRCNN-pose',
        backbone = 'HrNet32', 
        #note if inference need HrNet (64)
        dataset_path = None
        ):

        self.backbone_det = backbone_det
        backbone_det_path = os.path.join('./utils/models', self.backbone_det)
        ckpt_cfg_list = list(os.listdir(backbone_det_path))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.det_config = os.path.join(backbone_det_path, item)
            elif item[-1] == 'h':
                self.det_checkpoint = os.path.join(backbone_det_path, item)
            else:
                print("Warning!!! There is an unrecognized file in the backbone folder.")

        self.backbone = backbone
        backbone_path = os.path.join('./utils/models', self.backbone)
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


    def train(self, random_seed=0, save_fold='./checkpoints/pose_model/',checkpoint = None, distributed=False, validate=True,
              metric='PCK', save_best = 'PCK',optimizer="Adam", epochs=100, lr=5e-4):

        # self.cfg = Config.fromfile(self.backbonedict[self.backbone])
        # print(self.cfg.pretty_text)
        self.cfg.gpu_ids = range(1)
        self.cfg.work_dir = save_fold
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

        
    def inference(self, device='cpu',
                 pretrain_model = './checkpoints/pose_model/latest.pth',
                 is_trained=False,
                img=None, show=True,
                work_dir=None):
        print("========= begin inference ==========")
        
        if is_trained == True:
            self.pose_checkpoint = pretrain_model

        # initialize pose model
        pose_model = init_pose_model(self.pose_config, self.pose_checkpoint)
        # initialize detector
        det_model = init_detector(self.det_config, self.det_checkpoint)

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
        # reduce image size
        vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
        from IPython.display import Image, display
        import tempfile
        import os.path as osp
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = osp.join('./', 'pose_results.png')
            cv2.imwrite(file_name, vis_result)
            display(Image(file_name))
        return None


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
