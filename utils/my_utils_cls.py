from turtle import Turtle
import mmcv
import os.path as osp
from mmcv import Config
import time
from mmcls.apis import inference_model, init_model, show_result_pyplot, train_model
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset
from mmcv.runner import load_checkpoint
import os


class MMClassification:
    def __init__(self, 
        backbone='MobileNet',
        num_classes=-1
        # dataset_type = 'ImageNet'
        ):

        # self.backbone = backbone
        # 默认的config和checkpoints后续改为ResNet
        self.config = './utils/moodels/ResNet/ResNet50.py'
        self.checkpoint = './utils/moodels/ResNet/ResNet50.pth'

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
            "MobileNet": './utils/models/MobileNet/MobileNet.py',
            "ResNet": './utils/models/ResNet/ResNet50.py',
            'LeNet': './utils/models/LeNet/LeNet.py'
            # 下略
        }

        self.num_classes = num_classes
        # print(self.num_classes,"==================================\n")
        # self.dataset_type = dataset_type


    def train(self, random_seed=0, save_fold='./checkpoints', distributed=False, validate=True, device="cpu",
              metric='accuracy', optimizer="SGD", epochs=100, lr=0.001, weight_decay=0.001):# 加config

        # 获取config信息

        self.cfg = Config.fromfile(self.backbonedict[self.backbone])

        # print(self.cfg.model.backbone.keys())
        '''for lenet mnist, model.head
        for MobileNet, model.backbone
        crazy!!!!'''

        # model_head = list(self.cfg.model.keys())[1]
        # print(self.num_classes,"==================================\n")
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
        model = build_classifier(self.cfg.model)
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
        
    def inference(self, device='cpu',
                 pretrain_model = './checkpoints/latest.pth',
                 is_trained=False,
                image=None, show=True):
        print("========= begin inference ==========")
        model_fold = self.cfg.work_dir
        
        img_array = mmcv.imread(image)
        checkpoint = self.checkpoint
        if is_trained:
            checkpoint = pretrain_model
        model = init_model(self.cfg, checkpoint, device=device)
        result = inference_model(model, img_array) # 此处的model和外面的无关,纯局部变量
        if show == True:
            show_result_pyplot(model, image, result)
        return result

    def load_dataset(self, path):
        self.dataset_path = path

        self.cfg.img_norm_cfg = dict(
            mean=[124.508, 116.050, 106.438],
            std=[58.577, 57.310, 57.437],
            to_rgb=True
        )


        self.cfg.data.train.data_prefix = path + '/training_set/training_set'
        self.cfg.data.train.classes = path + '/classes.txt'
        # self.cfg.data.train.ann_file = path + '/train.txt'

        self.cfg.data.val.data_prefix = path + '/val_set/val_set'
        self.cfg.data.val.ann_file = path + '/val.txt'
        self.cfg.data.val.classes = path + '/classes.txt'

        self.cfg.data.test.data_prefix = path + '/test_set/test_set'
        self.cfg.data.test.ann_file = path + '/test.txt'
        self.cfg.data.test.classes = path + '/classes.txt'

    # def print_configs(self):
    #     if self.backbone is None:
    #         model = "MobileNet"
    #     else:
    #         model = self.backbone
    #     print("当前网络结构：" + model)
    #     print("数据集路径：", self.dataset_path)
    #     print("学习率", self.lr)

# if __name__ == "__main__":
#     img = '../img/test.jpg'
#     # mmcls_test(img)
#     model = MMClassification()
#     result = model.inference(image=img)
#     print(result)
#     show_result_pyplot(model.SOTA_model, img, result)
