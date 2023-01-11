import os
import json
import mmcv
import time
from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, train_detector
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmcv.runner import load_checkpoint
from tqdm import tqdm
import warnings
import torch
import numpy as np
import PIL
import cv2

warnings.filterwarnings("ignore")


class MMDetection:
    def sota(self):
        pypath = os.path.abspath(__file__)
        father = os.path.dirname(pypath)
        models = os.path.join(father, 'models')
        sota_model = []
        for i in os.listdir(models):
            if i[0] != '_':
                sota_model.append(i)
        return sota_model

    def __init__(self,
                 backbone='FasterRCNN',
                 num_classes=-1,
                 dataset_path=None
                 ):

        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.save_fold = None
        self.is_sample = False
        if backbone not in self.sota():
            if os.path.exists(backbone):  # 传入配置文件
                self.config = backbone
                self.cfg = Config.fromfile(self.config)
                self.backbone = backbone
            else:
                info = "Error Code: -302. No such argument: " + backbone + ". Currently " + str(
                    self.sota()) + " is available."
                # print(info)
                raise Exception(info)
        elif backbone in self.sota():
            self.config = os.path.join(
                self.file_dirname, 'models', 'FasterRCNN/FasterRCNN.py')
            self.checkpoint = os.path.join(
                self.file_dirname, 'models', '/FasterRCNN/FasterRCNN.pth')
            self.backbone = backbone
            backbone_path = os.path.join(
                self.file_dirname, 'models', self.backbone)
            ckpt_cfg_list = list(os.listdir(backbone_path))
            for item in ckpt_cfg_list:
                if item[-1] == 'y' and item[0] != '_':  # pip包修改1
                    self.config = os.path.join(backbone_path, item)
                elif item[-1] == 'h':
                    self.checkpoint = os.path.join(backbone_path, item)
                else:
                    # print("Warning!!! There is an unrecognized file in the backbone folder.")
                    pass
            self.cfg = Config.fromfile(self.config)

        self.dataset_path = dataset_path
        self.lr = None
        self.backbonedict = {
            "FasterRCNN": os.path.join(self.file_dirname, 'models', 'FasterRCNN/FasterRCNN.py'),
            "Yolov3": os.path.join(self.file_dirname, 'models', 'Yolov3/Yolov3.py'),
            # 下略
        }
        self.num_classes = num_classes
        self.chinese_res = None
        self.is_sample = False
        self.image_type = ""

    def train(self, random_seed=0, save_fold=None, distributed=False, validate=True, device='cpu',
              metric='bbox', save_best='bbox_mAP', optimizer="SGD", epochs=100, lr=0.001, weight_decay=0.001,
              Frozen_stages=1,
              checkpoint=None):

        # 加载网络模型的配置文件
        self.cfg = Config.fromfile(self.backbonedict[self.backbone])

        # 如果外部不指定save_fold
        if not self.save_fold:
            # 如果外部也没有传入save_fold，我们使用默认路径
            if not save_fold:
                self.save_fold = os.path.join(
                    self.cwd, 'checkpoints/det_model')
            # 如果外部传入save_fold，我们使用传入值
            else:
                self.save_fold = save_fold

        self.cfg.model.backbone.frozen_stages = Frozen_stages

        if self.num_classes != -1:
            self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes

        self.load_dataset(self.dataset_path)
        # 添加需要进行检测的类名
        self.cfg.classes = self.get_classes(self.cfg.data.train.ann_file)
        # 分别为训练、测试、验证添加类名
        self.cfg.data.train.classes = self.cfg.classes
        self.cfg.data.test.classes = self.cfg.classes
        self.cfg.data.val.classes = self.cfg.classes

        # 进行
        self.cfg.work_dir = self.save_fold
        # 创建工作目录
        mmcv.mkdir_or_exist(os.path.abspath(self.cfg.work_dir))
        # 创建分类器
        datasets = [build_dataset(self.cfg.data.train)]
        model = build_detector(self.cfg.model, train_cfg=self.cfg.get(
            'train_cfg'), test_cfg=self.cfg.get('test_cfg'))
        # print("checkpoint", checkpoint)
        if not checkpoint:
            model.init_weights()
        else:
            checkpoint = os.path.abspath(checkpoint)  # pip修改2
            load_checkpoint(model, checkpoint, map_location=torch.device(device))

        model.CLASSES = self.cfg.classes
        if optimizer == 'Adam':
            self.cfg.optimizer = dict(type='Adam', lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
        elif optimizer == 'Adagrad':
            self.cfg.optimizer = dict(type='Adagrad', lr=lr, lr_decay=0)
        # 根据输入参数更新config文件
        self.cfg.optimizer.lr = lr  # 学习率
        self.cfg.optimizer.type = optimizer  # 优化器
        self.cfg.optimizer.weight_decay = weight_decay  # 优化器的衰减权重
        self.cfg.evaluation.metric = metric  # 验证指标
        # self.cfg.evaluation.save_best = save_best
        self.cfg.runner.max_epochs = epochs  # 最大的训练轮次

        # 设置每 5 个训练批次输出一次日志
        # self.cfg.log_config.interval = 1
        self.cfg.gpu_ids = range(1)

        self.cfg.seed = random_seed

        train_detector(
            model,
            datasets,
            self.cfg,
            distributed=distributed,
            validate=validate,
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            meta=dict()
        )

    def print_result(self, res=None):
        if self.is_sample == True:
            print("示例检测结果如下：")
            sample_result = r"[{'类别标签': 0, '置信度': 1.0, '坐标': {'x': 26, 'y': 81, 'w': 497, 'h': 414}},{'类别标签': 1, '置信度': 1.0, '坐标': {'x': 250, 'y': 103, 'w': 494, 'h': 341}}]"
            print(sample_result)
        else:
            print("检测结果如下：")
            print(self.chinese_res)
        return self.chinese_res

    def load_checkpoint(self, checkpoint=None, class_path="../dataset/det/coco/classes.txt", device='cpu',
                        rpn_threshold=0.7, rcnn_threshold=0.7, **kwargs):
        # print("========= begin inference ==========")
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        if device not in ['cpu', 'cuda']:
            info = "Error Code: -301. No such argument: " + device
            raise Exception(info)
        is_cuda = torch.cuda.is_available()
        if device == 'cpu' and is_cuda:
            print("You can use  'device=cuda' to accelerate !")
        elif device == 'cuda' and not is_cuda:
            raise Exception("Error Code: -301. Your device doesn't support cuda.")
        if checkpoint != None and checkpoint.split(".")[-1] != 'pth':
            info = "Error Code: -202. Checkpoint file type error:" + checkpoint
            raise Exception(info)

        if self.num_classes != -1:
            self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes

        if checkpoint:
            # 加载数据集及配置文件的路径
            # self.load_dataset(self.dataset_path)
            # 修正检测的目标
            self.cfg.classes = self.get_class(class_path)
            self.cfg.data.train.classes = self.cfg.classes
            self.cfg.data.test.classes = self.cfg.classes
            self.cfg.data.val.classes = self.cfg.classes
            self.cfg.model.roi_head.bbox_head.num_classes = len(
                self.cfg.classes)
            self.checkpoint = checkpoint

        try:
            self.infer_model = init_detector(self.cfg, self.checkpoint, device=device)
        except FileNotFoundError:
            info = "Error Code: -102. No such checkpoint file:" + checkpoint
            raise Exception(info)
        self.infer_model.test_cfg.rpn.nms.iou_threshold = 1 - rpn_threshold
        self.infer_model.test_cfg.rcnn.nms.iou_threshold = 1 - rcnn_threshold

    def fast_inference(self, image=None, show=False, save_fold='det_result', **kwargs):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        try:
            self.infer_model
        except:
            print("请先使用load_checkpoint()方法加载权重！")
            return

        if type(image) != np.ndarray and image == None:
            self.is_sample = True
            sample_return = """
        [array([[ 26.547777  ,  81.55447   , 497.37015   , 414.4934    ,
          1.0]], dtype=float32), 
          array([[2.5098564e+02, 1.0334784e+02, 4.9422855e+02, 3.4187744e+02,
        1.0],  dtype=float32)]
            """
            return sample_return
        self.is_sample = False
        if isinstance(image, str) and not os.path.exists(image):
            info = "Error Code: -103. No such file:" + image
            raise Exception(info)
        if isinstance(image, str) and os.path.isfile(image) and image.split(".")[-1].lower() not in ["png", "jpg",
                                                                                                     "jpeg", "bmp"]:
            info = "Error Code: -203. File type error:" + image
            raise Exception(info)

        print("========= begin inference ==========")
        classed_name = self.infer_model.CLASSES
        self.num_classes = len(classed_name)

        results = []
        dataset_path = os.getcwd()
        img_input = []
        if not isinstance(image, str) and type(image) != np.ndarray and type(image) != list:  # 以PIL读入图片
            self.image_type = "pil"
            image = np.array(image)

        if type(image) == np.ndarray:
            if self.cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
                self.cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

            if self.image_type != "pil": self.image_type = "numpy"
            print("{} image".format(self.image_type))
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_input.append(mmcv.imread(image, channel_order='rgb'))
        elif type(image) == list:
            self.image_type = "list"
            if self.cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
                self.cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            for img in image:
                img_input.append(mmcv.imread(np.array(img)))
        elif os.path.isfile(image):
            self.image_type = "file"
            img_input.append(mmcv.imread(image, channel_order='rgb'))
        elif os.path.isdir(image):
            self.image_type = "path"
            for img in os.listdir(image):
                img_input.append(os.path.join(image, img))

        results = []
        # 推理
        for i, img in enumerate(tqdm(img_input)):
            results.append(inference_detector(self.infer_model, img))
        # 保存
        for i, img in enumerate(img_input):
            if type(img) != str: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.infer_model.show_result(img, results[i], show=show, out_file=os.path.join(save_fold,
                                                                                           "{}_img_{}.jpg".format(
                                                                                               self.image_type,
                                                                                               str(i))))

        chinese_res = []
        for result in results:
            for i in range(len(result)):
                for j in range(result[i].shape[0]):
                    tmp = {}
                    tmp['类别标签'] = i
                    tmp['置信度'] = result[i][j][4]
                    tmp['坐标'] = {"x": int(result[i][j][0]), "y": int(
                        result[i][j][1]), 'w': int(result[i][j][2]), 'h': int(result[i][j][3])}
                    # img.append(tmp)
                    chinese_res.append(tmp)
            # print(chinese_res)
        self.chinese_res = chinese_res
        print("========= finish inference =========")
        return results

    def inference(self, device='cpu',
                  checkpoint=None,
                  image=None,
                  show=True,
                  rpn_threshold=0.7,
                  rcnn_threshold=0.7,
                  class_path="../dataset/det/coco/classes.txt",
                  save_fold='det_result',
                  ):

        if checkpoint != None and checkpoint.split(".")[-1] != 'pth':
            info = "Error Code: -202. Checkpoint file type error:" + checkpoint
            raise Exception(info)

        checkpoint = os.path.abspath(checkpoint)  # pip修改2
        self.load_checkpoint(device=device, checkpoint=os.path.abspath(checkpoint), class_path=class_path,
                             rpn_threshold=rpn_threshold, rcnn_threshold=rcnn_threshold)
        return self.fast_inference(image=image, show=show, save_fold=save_fold)

    def load_dataset(self, path):
        self.dataset_path = path

        # 数据集修正为coco格式
        self.cfg.data.train.img_prefix = os.path.join(self.dataset_path, 'images/training_set/')
        self.cfg.data.train.ann_file = os.path.join(self.dataset_path, 'annotations/train.json')

        self.cfg.data.val.img_prefix = os.path.join(self.dataset_path, 'images/val_set/')
        self.cfg.data.val.ann_file = os.path.join(self.dataset_path, 'annotations/valid.json')

        self.cfg.data.test.img_prefix = os.path.join(self.dataset_path, 'images/test_set/')
        self.cfg.data.test.ann_file = os.path.join(self.dataset_path, 'annotations/test.json')

    def get_class(self, class_path):
        classes = []
        with open(class_path, 'r') as f:
            for name in f:
                classes.append(name.strip('\n'))
        return classes

    def get_classes(self, annotation_file):
        classes = ()
        with open(annotation_file, 'r') as f:
            dataset = json.load(f)
            # categories = dataset["categories"]
            if 'categories' in dataset:
                for cat in dataset['categories']:
                    classes = classes + (cat['name'],)
        return classes
