import os
import mmcv
import time
import torch
from mmcv import Config
from mmcls.apis import inference_model, init_model, show_result_pyplot, train_model, set_random_seed
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset
from mmcv.runner import load_checkpoint
from tqdm import tqdm


class MMClassification:
    def __init__(
            self,
            backbone='LeNet',
            num_classes=-1,
            dataset_path='../dataset/cls/hand_gray',
            # dataset_type = 'ImageNet'
    ):

        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.save_fold = None

        self.config = os.path.join(self.file_dirname, 'models', 'LeNet/LeNet.py')
        self.checkpoint = os.path.join(self.file_dirname, 'models', 'LeNet/LeNet.pth')

        self.backbone = backbone
        backbone_path = os.path.join(self.file_dirname, 'models', self.backbone)
        ckpt_cfg_list = list(os.listdir(backbone_path))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.config = os.path.join(backbone_path, item)
            elif item[-1] == 'h':
                self.checkpoint = os.path.join(backbone_path, item)
            else:
                #     print("Warning!!! There is an unrecognized file in the backbone folder.")
                pass

        self.cfg = Config.fromfile(self.config)
        self.dataset_path = dataset_path
        self.lr = None
        self.backbonedict = {
            "MobileNet": os.path.join(self.file_dirname, 'models', 'MobileNet/MobileNet.py'),
            "ResNet50": os.path.join(self.file_dirname, 'models', 'ResNet50/ResNet50.py'),
            "ResNet18": os.path.join(self.file_dirname, 'models', 'ResNet18/ResNet18.py'),
            "LeNet": os.path.join(self.file_dirname, 'models', 'LeNet/LeNet.py'),
            "RepVGG": os.path.join(self.file_dirname, 'models', 'RepVGG/RepVGG.py'),
            "RegNet": os.path.join(self.file_dirname, 'models', 'RegNet/RegNet.py'),
            "ResNeXt": os.path.join(self.file_dirname, 'models', 'ResNeXt/ResNeXt.py'),
            "VGG": os.path.join(self.file_dirname, 'models', 'VGG/VGG.py'),
            "ShuflleNet_v2": os.path.join(self.file_dirname, 'models', 'ShuflleNet_v2/ShuflleNet_v2.py'),
            # 下略
        }

        self.num_classes = num_classes
        self.chinese_res = None

    def train(self, random_seed=0, save_fold=None, distributed=False, validate=True, device="cuda",
              metric='accuracy', save_best='auto', optimizer="SGD", epochs=100, lr=0.01, weight_decay=0.001,
              checkpoint=None, evaluation_interval = 5):
        set_random_seed(seed=random_seed)
        # 获取config信息
        self.cfg = Config.fromfile(self.backbonedict[self.backbone])

        # 如果外部不指定save_fold
        if not self.save_fold:
            # 如果外部也没有传入save_fold，我们使用默认路径
            if not save_fold:
                self.save_fold = os.path.join(self.cwd, 'checkpoints/cls_model')
            # 如果外部传入save_fold，我们使用传入值
            else:
                self.save_fold = save_fold

        if self.num_classes != -1:
            if 'num_classes' in self.cfg.model.backbone.keys():
                self.cfg.model.backbone.num_classes = self.num_classes
            else:
                self.cfg.model.head.num_classes = self.num_classes

        self.load_dataset(self.dataset_path)

        # 进行
        self.cfg.work_dir = self.save_fold
        # 创建工作目录
        mmcv.mkdir_or_exist(os.path.abspath(self.cfg.work_dir))
        # 创建分类器
        model = build_classifier(self.cfg.model)
        if not checkpoint:
            model.init_weights()
        else:
            load_checkpoint(model, checkpoint, map_location=torch.device('cpu'))
            # model = init_model(self.cfg, checkpoint)

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
        if optimizer == 'Adam':
            self.cfg.optimizer = dict(type='Adam', lr=lr,betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0001)
        elif optimizer == 'Adagrad':
            self.cfg.optimizer = dict(type='Adagrad',lr=lr, lr_decay=0)
        self.cfg.optimizer.weight_decay = weight_decay  # 优化器的衰减权重
        self.cfg.evaluation.metric = metric  # 验证指标
        self.cfg.evaluation.interval = evaluation_interval # 验证间隔
        self.cfg.evaluation.save_best = save_best  #
        self.cfg.runner.max_epochs = epochs  # 最大的训练轮次

        # 设置每 5 个训练批次输出一次日志
        self.cfg.log_config.interval = 10
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

    def print_result(self, res=None):
        print("检测结果如下：")
        print(self.chinese_res)
        return self.chinese_res

    def inference(self, device='cpu',
                  checkpoint=None,
                  image=None,
                  show=True,
                  class_path="../dataset/classes/cls_classes.txt",
                  save_fold='cls_result',
                  encoding='utf-8'
                  ):

        if not checkpoint:
            checkpoint = os.path.join(self.cwd, 'checkpoints/cls_model/hand_gray/latest.pth')

        print("========= begin inference ==========")
        classed_name = self.get_class(class_path, encoding)
        self.num_classes = len(classed_name)

        if self.num_classes != -1:
            if 'num_classes' in self.cfg.model.backbone.keys():
                self.cfg.model.backbone.num_classes = self.num_classes
            else:
                self.cfg.model.head.num_classes = self.num_classes
        model = init_model(self.cfg, checkpoint, device=device)
        model.CLASSES = classed_name
        results = []
        if (image[-1] != '/'):
            img_array = mmcv.imread(image, flag='grayscale' if self.backbone == "LeNet" else 'color')
            result = inference_model(model, img_array)  # 此处的model和外面的无关,纯局部变量
            #if show == True:
            #    show_result_pyplot(model, image, result)
            model.show_result(image, result, show=show, out_file=os.path.join(save_fold, image))
            chinese_res = []
            tmp = {}
            tmp['标签'] = result['pred_label']
            tmp['置信度'] = result['pred_score']
            tmp['预测结果'] = result['pred_class']
            # img.append(tmp)
            chinese_res.append(tmp)
            # print(chinese_res)
            self.chinese_res = chinese_res
            print("========= finish inference ==========")
            return result
        else:
            img_dir = image
            mmcv.mkdir_or_exist(os.path.abspath(save_fold))
            chinese_results = []
            for i, img in enumerate(tqdm(os.listdir(img_dir))):
                result = inference_model(model, img_dir + img)  # 此处的model和外面的无关,纯局部变量
                model.show_result(img_dir + img, result, out_file=os.path.join(save_fold, img))
                chinese_res = []
                chinese_res = []
                tmp = {}
                tmp['标签'] = result['pred_label']
                tmp['置信度'] = result['pred_score']
                tmp['预测结果'] = result['pred_class']
                # img.append(tmp)
                chinese_res.append(tmp)
                chinese_results.append(chinese_res)
                results.append(result)
            self.chinese_res = chinese_results

        print("========= finish inference ==========")

        return results

    def load_dataset(self, path):
        self.dataset_path = path

        self.cfg.img_norm_cfg = dict(
            mean=[124.508, 116.050, 106.438],
            std=[58.577, 57.310, 57.437],
            to_rgb=True
        )

        self.cfg.data.train.data_prefix = os.path.join(self.dataset_path, 'training_set')
        self.cfg.data.train.classes = os.path.join(self.dataset_path, 'classes.txt')

        self.cfg.data.val.data_prefix = os.path.join(self.dataset_path, 'val_set')
        self.cfg.data.val.ann_file = os.path.join(self.dataset_path, 'val.txt')
        self.cfg.data.val.classes = os.path.join(self.dataset_path, 'classes.txt')

        self.cfg.data.test.data_prefix = os.path.join(self.dataset_path, 'test_set')
        self.cfg.data.test.ann_file = os.path.join(self.dataset_path, 'test.txt')
        self.cfg.data.test.classes = os.path.join(self.dataset_path, 'classes.txt')

    def get_class(self, class_path, encoding = 'utf-8'):
        classes = []
        with open(class_path, 'r', encoding = encoding) as f:
            for name in f:
                classes.append(name.strip('\n'))
        return classes