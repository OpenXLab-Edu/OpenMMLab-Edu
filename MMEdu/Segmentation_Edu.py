import mmcv
import os.path as osp
from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, set_random_seed, train_segmentor
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
import time
import os
import json
import cv2
from PIL import Image
import numpy as np


class MMSegmentation:
    def __init__(self, 
        backbone='UNet',
        num_classes=-1,
        dataset_path = None
        ):

        self.config = '../MMEdu/models/UNet/UNet.py'
        self.checkpoint = '../MMEdu/models/UNet/UNet.pth'

        self.backbone = backbone
        backbone_path = os.path.join('../MMEdu/models', self.backbone)
        ckpt_cfg_list = list(os.listdir(backbone_path))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.config = os.path.join(backbone_path, item)
            elif item[-1] == 'h':
                self.checkpoint = os.path.join(backbone_path, item)
            else:
                print("Warning!!! There is an unrecognized file in the backbone folder.")

        self.cfg = Config.fromfile(self.config)

        self.dataset_path = dataset_path
        self.lr = None
        self.backbonedict = {
            "UNet": '../MMEdu/models/UNet/UNet.py'
            # 下略
        }
        self.num_classes = num_classes


    def train(self, checkpoint = None, save_fold='./checkpoints', distributed=False, validate=True,
              metric='mIoU', optimizer="SGD", epochs=40000, lr=0.01, weight_decay=0.0005, 
              log_interval=5, eval_interval=100, random_seed=0):# 加config
        # 加载网络模型的配置文件
        self.cfg = Config.fromfile(self.backbonedict[self.backbone])

        if self.num_classes != -1:
            self.cfg.model.decode_head.num_classes = self.num_classes
            self.cfg.model.auxiliary_head.num_classes = self.num_classes

        self.load_dataset(self.dataset_path)

        print("进行了cfg的切换")
        # 进行
        self.cfg.work_dir = save_fold # self.save_fold
        self.cfg.optimizer.lr = lr  # 学习率
        self.cfg.optimizer.type = optimizer  # 优化器
        self.cfg.optimizer.weight_decay = weight_decay  # 优化器的衰减权重
        self.cfg.evaluation.metric = metric  # 验证指标
        self.cfg.evaluation.interval = eval_interval # 验证轮数
        self.cfg.runner.max_iters = epochs  # 最大的训练轮次
        self.cfg.log_config.interval = log_interval # 日志输出轮次
        self.cfg.gpu_ids = range(1)
        self.cfg.seed = random_seed
        set_random_seed(0, deterministic=False)

        # 创建工作目录
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))

        # 创建分类器
        datasets = [build_dataset(self.cfg.data.train)]

        # add CLASSES and PALETTE to checkpoint
        self.cfg.checkpoint_config.meta = dict(PALETTE=datasets[0].PALETTE)
        model = build_segmentor(self.cfg.model, train_cfg=self.cfg.get('train_cfg'), test_cfg=self.cfg.get('test_cfg'))
        model.CLASSES = datasets[0].CLASSES
        model.PALETTE = datasets[0].PALETTE

        if not checkpoint:
            model.init_weights()
        else:
            load_checkpoint(model, checkpoint)

        train_segmentor(
            model,
            datasets,
            self.cfg,
            distributed=distributed,
            validate=validate,
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            meta=dict()
        )

        
    def inference(self, device='cpu',
                 pretrain_model = './checkpoints/latest.pth',
                 is_trained=False,
                 image=None, show=True,
                 work_dir='../checkpoints'):
        print("========= begin inference ==========")
        #model_fold = self.cfg.work_dir
        if self.num_classes != -1:
            self.cfg.model.decode_head.num_classes = self.num_classes
            self.cfg.model.auxiliary_head.num_classes = self.num_classes

        img_array = mmcv.imread(image)
        checkpoint = self.checkpoint

        # config = self.cfg
        #self.cfg.model.pretrained = None
        #self.cfg.model.train_cfg = None
        model = build_segmentor(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))

        if is_trained:
            # 加载数据集及配置文件的路径
            checkpoint = pretrain_model
            self.load_dataset(self.dataset_path)
            # 修正检测的目标
            self.cfg.classes = self.get_classes()
            self.cfg.palette = self.get_palette()
            self.cfg.data.train.classes = self.cfg.classes
            self.cfg.data.test.classes = self.cfg.classes
            self.cfg.data.val.classes = self.cfg.classes
            self.cfg.data.train.palette = self.cfg.palette
            self.cfg.data.test.palette = self.cfg.palette
            self.cfg.data.val.palette = self.cfg.palette
            model.CLASSES = self.cfg.classes
            model.PALETTE = self.cfg.palette

        if checkpoint is not None:

            load_checkpoint(model, checkpoint, map_location='cpu')
        model = init_segmentor(self.cfg, checkpoint, device=device)

        result = inference_segmentor(model, img_array) # 此处的model和外面的无关,纯局部变量
        if show == True:
            show_result_pyplot(model, image, result)
            # cv2.imwrite('demo/city_seg.jpg', result[0])
        return result


    def load_dataset(self, path, dataset_type='StandfordBackgroundDataset'):

        self.dataset_path = path
        self.cfg.data.train.img_dir = os.path.join(path, 'images/')
        self.cfg.data.train.ann_dir = os.path.join(path, 'labels/')
        self.cfg.data.train.split = os.path.join(path, 'splits/val.txt')

        self.cfg.data.val.img_dir = os.path.join(path, 'images/')
        self.cfg.data.val.ann_dir = os.path.join(path, 'labels/')
        self.cfg.data.val.split = os.path.join(path, 'splits/val.txt')

        self.cfg.data.test.img_dir = os.path.join(path, 'images/')
        self.cfg.data.test.ann_dir = os.path.join(path, 'labels/')
        self.cfg.data.test.split = os.path.join(path, 'splits/val.txt')


    def get_classes(self):
        classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
        return classes


    def get_palette(self):
        palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
            [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]
        return palette


    def data(self, data_path, ann_dir):
        data_path = 'data/'
        ann_dir = 'labels/'
        classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
        palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
                [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

        for file in mmcv.scandir(osp.join(data_path, ann_dir), suffix='.regions.txt'):
            seg_map = np.loadtxt(osp.join(data_path, ann_dir, file)).astype(np.uint8)
            seg_img = Image.fromarray(seg_map).convert('P')
            seg_img.putpalette(np.array(palette, dtype=np.uint8))
            seg_img.save(osp.join(data_path, ann_dir, file.replace('.regions.txt',
                                                                    '.png')))

        # split train/val set randomly
        split_dir = 'splits'
        mmcv.mkdir_or_exist(osp.join(data_path, split_dir))
        filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
            osp.join(data_path, ann_dir), suffix='.png')]
        with open(osp.join(data_path, split_dir, 'train.txt'), 'w') as f:
        # select first 4/5 as train set
            train_length = int(len(filename_list)*4/5)
            f.writelines(line + '\n' for line in filename_list[:train_length])
        with open(osp.join(data_path, split_dir, 'val.txt'), 'w') as f:
        # select last 1/5 as train set
            f.writelines(line + '\n' for line in filename_list[train_length:])

        # config = './utils/models/PSPNet/PSPNet.py'
        # cfg = Config.fromfile(config)
        self.cfg.classes = classes
        self.cfg.palette = palette