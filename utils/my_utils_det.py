import mmcv
import os.path as osp
from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, train_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmcv.runner import load_checkpoint
import os
'''
TODO:
    1.数据集格式coco,路径修正
    2.
'''


class MMDetection:
    def __init__(self, 
        backbone='FasterRCNN',
        num_classes=-1
        # dataset_type = 'ImageNet'
        ):

        self.config = './utils/models/FasterRCNN/FasterRCNN.py'
        self.checkpoint = './utils/models/FasterRCNN/FasterRCNN.pth'

        self.backbone = backbone
        backbone_path = os.path.join('./utils/models', self.backbone)
        ckpt_cfg_list = list(os.listdir(backbone_path))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.config = os.path.join(backbone_path, item)
            elif item[-1] == 'h':
                self.checkpoint = os.path.join(backbone_path, item)
            else:
                print("Warning!!! There is an unrecognized file in the backbone folder.")

        self.cfg = Config.fromfile(self.config)

        self.dataset_path = None
        self.lr = None
        self.backbonedict = {
            "FasterRCNN": './utils/models/MobileNet/MobileNet.py',
            "Yolov3": './utils/models/ResNet/ResNet50.py',
            # 下略
        }

        self.num_classes = num_classes
        return None


    def train(self, random_seed=0, save_fold='./checkpoints', distributed=False, validate=True, device="cpu",
              metric='mAP', optimizer="SGD", epochs=100, lr=0.001, weight_decay=0.001, Frozen_stages=1):# 加config

        self.cfg = Config.fromfile(self.backbonedict[self.backbone])

        self.cfg.model.backbone.frozen_stages = Frozen_stages

        if self.num_classes != -1:
            if 'num_classes' in self.cfg.model.backbone.keys():
                self.cfg.model.backbone.num_classes = self.num_classes
            else:
                self.cfg.model.head.num_classes = self.num_classes

        self.load_dataset(self.dataset_path)
        print("进行了cfg的切换")
            # 进行
        self.cfg.work_dir = save_fold
        # 创建工作目录
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
        # 创建分类器
        model = build_detector(self.cfg.model)
        model.init_weights()

        datasets = [build_dataset(self.cfg.data.train)]

        # 添加类别属性以方便可视化
        model.CLASSES = datasets[0].CLASSES

        n_class = len(model.CLASSES) 
        if n_class <= 5:
            self.cfg.evaluation.metric_options = {'topk': (1,)}
        else:
            self.cfg.evaluation.metric_options = {'topk': (5,)}

        # 根据输入参数更新config文件
        self.cfg.optimizer.lr = lr  # 学习率
        self.cfg.optimizer.type = optimizer  # 优化器
        self.cfg.optimizer.weight_decay = weight_decay  # 优化器的衰减权重
        self.cfg.evaluation.metric = metric  # 验证指标
        self.cfg.runner.max_epochs = epochs  # 最大的训练轮次

        # 设置每 5 个训练批次输出一次日志
        self.cfg.log_config.interval = 1
        self.cfg.gpu_ids = range(1)

        self.cfg.seed = random_seed

        train_model(
            model,
            datasets,
            self.cfg,
            distributed=distributed,
            validate=validate,
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            device=device,
            meta=dict()
        )

        return None

        
    def inference(self, device='cpu',
                 pretrain_model = './checkpoints/latest.pth',
                 is_trained=False,
                infer_data=None, show=True, iou_threshold=0.9):
        print("========= begin inference ==========")
        model_fold = self.cfg.work_dir
        
        img_array = mmcv.imread(infer_data)
        checkpoint = self.checkpoint
        if is_trained:
            checkpoint = pretrain_model
        model = init_detector(self.cfg, checkpoint, device=device)
        result = inference_detector(model, img_array) # 此处的model和外面的无关,纯局部变量
        if show == True:
            show_result_pyplot(model, infer_data, result)
        return result


    def load_dataset(self, path, dataset_type=='CocoDataset'):

        self.dataset_type = dataset_type
        self.dataset_path = path

        #数据集修正为coco格式

        self.cfg.data.train.data_prefix = path + '/training_set/training_set'
        self.cfg.data.train.classes = path + '/classes.txt'
        # self.cfg.data.train.ann_file = path + '/train.txt'

        self.cfg.data.val.data_prefix = path + '/val_set/val_set'
        self.cfg.data.val.ann_file = path + '/val.txt'
        self.cfg.data.val.classes = path + '/classes.txt'

        self.cfg.data.test.data_prefix = path + '/test_set/test_set'
        self.cfg.data.test.ann_file = path + '/test.txt'
        self.cfg.data.test.classes = path + '/classes.txt'
