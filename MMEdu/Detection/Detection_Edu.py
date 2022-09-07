import os
import mmcv
import time
import torch
from mmcv import Config
from mmcls.apis import inference_model, init_model, show_result_pyplot, train_model, set_random_seed, single_gpu_test
from mmcls.models import build_classifier
from mmcls.datasets import  build_dataloader,build_dataset
from mmcv.runner import load_checkpoint
from tqdm import tqdm
import numpy as np


class MMClassification:
    def sota():
        pypath = os.path.abspath(__file__)
        father = os.path.dirname(pypath)
        models = os.path.join(father, 'models')
        sota_model = []
        for i in os.listdir(models):
            if i[0] != '_':
                sota_model.append(i)
        return sota_model

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
            if item[-1] == 'y' and item[0] != '_':  #pip修改1
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
            'MobileNet': os.path.join(self.file_dirname, 'models', 'MobileNet/MobileNet.py'),
            'ResNet18': os.path.join(self.file_dirname, 'models', 'ResNet18/ResNet18.py'),
            'ResNet50': os.path.join(self.file_dirname, 'models', 'ResNet50/ResNet50.py'),
            'LeNet': os.path.join(self.file_dirname, 'models', 'LeNet/LeNet.py'),
            # 下略
        }

        self.num_classes = num_classes
        self.chinese_res = None

    def train(self, random_seed=0, save_fold=None, distributed=False, validate=True, device="cpu",
              metric='accuracy', save_best='auto', optimizer="SGD", epochs=100, lr=0.01, weight_decay=0.001,
              checkpoint=None):
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
        if optimizer == 'Adam':
            self.cfg.optimizer = dict(type='Adam', lr=lr,betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0001)
        elif optimizer == 'Adagrad':
            self.cfg.optimizer = dict(type='Adagrad',lr=lr, lr_decay=0)
        # 根据输入参数更新config文件
        self.cfg.optimizer.lr = lr  # 学习率
        self.cfg.optimizer.type = optimizer  # 优化器
        self.cfg.optimizer.weight_decay = weight_decay  # 优化器的衰减权重
        self.cfg.evaluation.metric = metric  # 验证指标
        self.cfg.evaluation.save_best = save_best  #
        self.cfg.runner.max_epochs = epochs  # 最大的训练轮次

        # 设置每 10 个训练批次输出一次日志
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
                  save_fold='cls_result'
                  ):

        if not checkpoint:
            checkpoint = os.path.join(self.cwd, 'checkpoints/cls_model/hand_gray/latest.pth')

        print("========= begin inference ==========")
        classed_name = self.get_class(class_path)
        self.num_classes = len(classed_name)

        if self.num_classes != -1:
            if 'num_classes' in self.cfg.model.backbone.keys():
                self.cfg.model.backbone.num_classes = self.num_classes
            else:
                self.cfg.model.head.num_classes = self.num_classes

        checkpoint = os.path.abspath(checkpoint) # pip修改2

        
        results = []
        if (image[-1] != '/'):
            if self.backbone != "LeNet":
                model = init_model(self.cfg, checkpoint, device=device)
                model.CLASSES = classed_name
                img_array = mmcv.imread(image, flag='color')
                result = inference_model(model, img_array)  # 此处的model和外面的无关,纯局部变量
            else: 
                # build the dataloader
                dataset_path = os.getcwd()
                f = open("test.txt",'w')
                f.write(image)
                f.write(" 1")
                f.write('\n')
                f.write("no.png 0")
                f.close()
                if not os.path.exists("test_set"):
                    os.mkdir('test_set')
                import shutil
                if not os.path.exists(image):
                    shutil.copyfile(image, os.path.join("test_set", image))
                shutil.copyfile(image, os.path.join("test_set", "no.png"))
                self.cfg.data.test.data_prefix = os.path.join(dataset_path,'test_set')
                self.cfg.data.test.ann_file = os.path.join(dataset_path,'test.txt')
                self.cfg.data.test.classes = os.path.abspath(class_path)

                dataset = build_dataset(self.cfg.data.test)
                # the extra round_up data will be removed during gpu/cpu collect
                data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=self.cfg.data.samples_per_gpu,
                    workers_per_gpu=self.cfg.data.workers_per_gpu,
                    shuffle=False,
                    round_up=True)
                model = build_classifier(self.cfg.model)
                checkpoint = load_checkpoint(model, checkpoint)
                result = single_gpu_test(model,data_loader )

                f = open(class_path, "r")
                ff = f.readlines()
                f.close()
                # print("\n",np.argmax(result[0]), ff[np.argmax(result[0])][-1:])
                pred_class = ff[np.argmax(result[0])] if ff[np.argmax(result[0])][-1:] != "\n" else ff[np.argmax(result[0])][:-1]
                result = {
                    'pred_label':np.argmax(result[0]),
                    'pred_score':result[0][np.argmax(result[0])],
                    'pred_class':pred_class,
                }
            model.show_result(image, result, show=show, out_file=os.path.join(save_fold, os.path.split(image)[1]))
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
            model = init_model(self.cfg, checkpoint, device=device)
            model.CLASSES = classed_name
            img_dir = image
            mmcv.mkdir_or_exist(os.path.abspath(save_fold))
            chinese_results = []
            for i, img in enumerate(tqdm(os.listdir(img_dir))):
                result = inference_model(model, img_dir + img)  # 此处的model和外面的无关,纯局部变量
                model.show_result(img_dir + img, result, out_file=os.path.join(save_fold, os.path.split(img)[1]))
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

    def get_class(self, class_path):
        classes = []
        with open(class_path, 'r') as f:
            for name in f:
                classes.append(name.strip('\n'))
        return classes
