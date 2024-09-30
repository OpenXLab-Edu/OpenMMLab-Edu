import mmcv
from mmcv import Config
import mmpose
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,train_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
import cv2
import os
import tempfile
import os.path as osp
from tqdm import tqdm

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
        self.save_fold = None
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

        self.cfg_det = Config.fromfile(self.det_config)
        self.cfg = Config.fromfile(self.pose_config)
        
        self.dataset_path = dataset_path

        return None


    def train(self, random_seed=0, save_fold=None, checkpoint = None, distributed=False, validate=True,
              metric='PCK', save_best = 'PCK',optimizer="Adam", epochs=100, lr=5e-4,
              resume_from = None,
              eval_interval = 10,
              log_interval = 5,
              ):
        print("========= begin training ==========")
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
        self.cfg.resume_from = resume_from
        self.cfg.seed = random_seed

        self.cfg.evaluation.interval = eval_interval
        self.cfg.evaluation.metric = metric  # 验证指标
        self.cfg.evaluation.save_best = save_best  # 验证指标
    

        # self.cfg.model.backbone.frozen_stages = Frozen_stages
        # set log interval
        self.cfg.log_config.interval = log_interval
        self.cfg.total_epochs = epochs  # 最大的训练轮次
        self.cfg.optimizer.lr = lr  # 学习率
        self.cfg.optimizer.type = optimizer  # 优化器

        datasets = [build_dataset(self.cfg.data.train)]

        # build model
        model = build_posenet(self.cfg.model)

        # create work_dir
        mmcv.mkdir_or_exist(self.cfg.work_dir)

        # train model
        train_model(
            model, datasets, self.cfg, distributed=distributed, validate=validate, meta=dict())
        print("========= finish training ==========")
        return None

    def _inference(self,det_model,pose_model,img,work_dir,name,show,i):
        mmdet_results = inference_detector(det_model, img)
        person_results = process_mmdet_results(mmdet_results, cat_id=1)
        pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                                img,
                                                                person_results,
                                                                bbox_thr=0.3,
                                                                format='xyxy',
                                                                dataset=pose_model.cfg.data.test.type)
        vis_result = vis_pose_result(pose_model,
                                img,
                                pose_results,
                                dataset=pose_model.cfg.data.test.type,
                                show=show)
        with tempfile.TemporaryDirectory() as tmpdir:
            if not os.path.exists(work_dir):   ##目录存在，返回为真
                os.makedirs(work_dir) 

            file_name = osp.join(work_dir, name+str(i)+'.png')
            cv2.imwrite(file_name, vis_result)
        return pose_results

    def inference(self,
                  device='cuda:0',
                  is_trained=False,
                  pretrain_model='./checkpoints/pose_model/latest.pth',
                  img=None,
                  show=False,
                  work_dir='./img_result/',
                  name='pose_result'):
        """
        params:
            device: 推理设备,可选参数: ('cuda:int','cpu')
            is_trained: 是否使用本地预训练的其他模型进行训练
            pretrain_model: 如果使用其他模型，则传入模型路径
            img: 推理图片的路径或文件夹名
            show: 是否对推理结果进行显示
            work_dir: 推理结果图片的保存文件夹
            name: 推理结果保存的名字
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


        # inference img
        if img[-1:] != '/':
            pose_results = self._inference(det_model,pose_model,img,work_dir,name,show,0)
            print('Image result is save as %s.png' % (name))

        else:
        # inference directory
            img_dir = img
            print("inference for directory: %s \n" % (img_dir))
            for i,img in enumerate(tqdm(os.listdir(img_dir))):
                pose_results = self._inference(det_model,pose_model,img_dir+img,work_dir,name,show,i)
            print('Finish! Image result is save in %s \n' % (work_dir))
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
