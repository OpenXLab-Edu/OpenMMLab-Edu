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
import cv2


class MMClassification:
    def sota(self):
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
            **kwargs,
    ):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))
        self.save_fold = None
        if backbone not in self.sota():
            if os.path.exists(backbone): # 传入配置文件
                self.config = backbone
                self.cfg = Config.fromfile(self.config)
                self.backbone = backbone
            else:
                info = "Error Code: -302. No such argument: "+backbone +". Currently "+str(self.sota())+" is available."
                # print(info)
                raise Exception(info)
        elif backbone in self.sota():

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
        self.is_sample = False

    def train(self, random_seed=0, save_fold=None, distributed=False, validate=True, device="cpu",
              metric='accuracy', save_best='auto', optimizer="SGD", epochs=100, lr=0.01, weight_decay=0.001,
              checkpoint=None,**kwargs):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        if device not in ['cpu','cuda']:
            info = "Error Code: -301. No such argument: "+ device
            raise Exception(info)
        if validate not in [True, False]:
            info = "Error Code: -303. No such argument: "+ validate
            raise Exception(info)
        if checkpoint != None and checkpoint.split(".")[-1] != 'pth':
            info = "Error Code: -202. Checkpoint file type error:"+ checkpoint
            raise Exception(info)

        set_random_seed(seed=random_seed)
        # 获取config信息
        if self.backbone.split('.')[-1] == 'py':
            self.cfg = Config.fromfile(self.backbone)
        else:
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

        datasets = None
        try:
            datasets = [build_dataset(self.cfg.data.train)]
        except FileNotFoundError as err:
            if not os.path.exists(self.dataset_path):
                info = "Error Code: -101. No such dataset directory:" + self.dataset_path
            else:
                err = str(err).split(":")[-1]
                info = "Error Code: -201. Dataset file type error.  No such file:"+ err
            raise Exception(info)
        # 进行
        self.cfg.work_dir = self.save_fold
        # 创建工作目录
        mmcv.mkdir_or_exist(os.path.abspath(self.cfg.work_dir))
        # 创建分类器
        model = build_classifier(self.cfg.model)
        if not checkpoint:
            model.init_weights()
        else:
            try:
                load_checkpoint(model, checkpoint, map_location=torch.device(device))
                # model = init_model(self.cfg, checkpoint)
            except FileNotFoundError:
                    info = "Error Code: -102. No such checkpoint file:" + checkpoint
                    raise Exception(info)


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
        self.cfg.device = device

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
        if self.is_sample == True:
            print("示例检测结果如下：")
            sample_result = r"[{'标签': 2, '置信度': 1.0, '预测结果': 'scissors'}]"
            print(sample_result)
        else:
            print("检测结果如下：")
            print(self.chinese_res)
        return self.chinese_res


    def load_checkpoint(self, device='cpu',
                  checkpoint=None,
                  class_path="../dataset/classes/cls_classes.txt",
                  **kwargs,
                  ):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: "+ next(iter(kwargs.keys()))
            raise Exception(info)
        if device not in ['cpu','cuda']:
            info = "Error Code: -301. No such argument: "+ device
            raise Exception(info)
        if checkpoint != None and checkpoint.split(".")[-1] != 'pth':
            info = "Error Code: -202. Checkpoint file type error:"+ checkpoint
            raise Exception(info)
        # if not checkpoint:
            # checkpoint = os.path.join(self.cwd, 'checkpoints/cls_model/hand_gray/latest.pth')

        print("========= begin inference ==========")
        classed_name = self.get_class(class_path)
        self.num_classes = len(classed_name)
        if self.num_classes != -1:
            if 'num_classes' in self.cfg.model.backbone.keys():
                self.cfg.model.backbone.num_classes = self.num_classes
            else:
                self.cfg.model.head.num_classes = self.num_classes

        checkpoint = os.path.abspath(checkpoint) # pip修改2
        try:
            self.infer_model = init_model(self.cfg, checkpoint, device=device)
        except FileNotFoundError:
            info = "Error Code: -102. No such checkpoint file:"+ checkpoint
            raise Exception(info)
        self.infer_model.CLASSES = classed_name

    def fast_inference(self, image, show=False, save_fold='cls_result',**kwargs):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        import PIL
        if type(image) == PIL.PngImagePlugin.PngImageFile:
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_array = mmcv.imread(image, flag='color')
        try:
            self.infer_model
        except:
            print("请先使用load_checkpoint()方法加载权重！")
            return 
        result = inference_model(self.infer_model, img_array)  # 此处的model和外面的无关,纯局部变量
        # self.infer_model.show_result(image, result, show=show, out_file=os.path.join(save_fold, os.path.split(image)[1]))
        if show:
            show_result_pyplot(self.infer_model, image, result)
        chinese_res = []
        tmp = {}
        tmp['标签'] = result['pred_label']
        tmp['置信度'] = result['pred_score']
        tmp['预测结果'] = result['pred_class']
        # img.append(tmp)
        chinese_res.append(tmp)
        # print(chinese_res)
        self.chinese_res = chinese_res
        # print("========= finish inference ==========")
        return result

    def inference(self, device='cpu',
        checkpoint=None,
        image=None,
        show=True,
        class_path="../dataset/classes/cls_classes.txt",
        save_fold='cls_result',
        **kwargs,
        ):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        if device not in ['cpu','cuda']:
            info = "Error Code: -301. No such argument: "+ device
            raise Exception(info)
        if image == None: # 传入图片为空，示例输出
            self.is_sample = True
            sample_return = """
{'pred_label': 2, 'pred_score': 0.9930743, 'pred_class': 'scissors'}
            """
            return sample_return
        self.is_sample = False
        # if not isinstance(image,(str, np.array)):
        if not isinstance(image,str): # 传入图片格式，仅支持str图片路径
            info = "Error Code: -304. No such argument:"+ image+"which is" +type(image)
            raise Exception(info)
        if not os.path.exists(image):
            info = "Error Code: -103. No such file:"+ image
            raise Exception(info)
        if image[-1] != '/' and image.split(".")[-1].lower() not in ["png","jpg","jpeg","bmp"]:
            info = "Error Code: -203. File type error:"+ image
            raise Exception(info)

        if checkpoint != None and checkpoint.split(".")[-1] != 'pth':
            info = "Error Code: -202. Checkpoint file type error:"+ checkpoint
            raise Exception(info)
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
                imagename = image.split("/")[-1]

                # build the dataloader
                dataset_path = os.getcwd()
                f = open("test.txt",'w')
                f.write(imagename)
                f.write(" 1")
                f.write('\n')
                f.write("no.png 0")
                f.close()
                if not os.path.exists("cache"):
                    os.mkdir('cache')
                import shutil
                if not os.path.exists(os.path.join("cache", imagename)):
                    shutil.copyfile(image, os.path.join("cache", imagename))
                shutil.copyfile(image, os.path.join("cache", "no.png"))
                self.cfg.data.test.data_prefix = os.path.join(dataset_path,'cache')
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
                try:
                    checkpoint = load_checkpoint(model, checkpoint)
                except FileNotFoundError:
                    info = "Error Code: -102. No such checkpoint file:"+ checkpoint
                    raise Exception(info)
                result = single_gpu_test(model,data_loader )
                os.remove("test.txt")
                shutil.rmtree("cache")
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
            if isinstance(result['pred_label'], np.int64):
                result['pred_label'] = int(result['pred_label'])
            if isinstance( result['pred_score'], np.float32):
                result['pred_score'] = float( result['pred_score'])
            tmp['标签'] = result['pred_label']
            tmp['置信度'] = result['pred_score']
            tmp['预测结果'] = result['pred_class']
            # img.append(tmp)
            chinese_res.append(tmp)
            # print(chinese_res)
            self.chinese_res = chinese_res
            print("\n========= finish inference ==========")
            return result
        else:
            if self.backbone != "LeNet":
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
                    if isinstance(result['pred_label'], np.int64):
                        result['pred_label'] = int(result['pred_label'])
                    if isinstance( result['pred_score'], np.float32):
                        result['pred_score'] = float( result['pred_score'])
                    tmp['标签'] = result['pred_label']
                    tmp['置信度'] = result['pred_score']
                    tmp['预测结果'] = result['pred_class']
                    # img.append(tmp)
                    chinese_res.append(tmp)
                    chinese_results.append(chinese_res)
                    results.append(result)
                self.chinese_res = chinese_results

                print("========= finish inference ==========")
            else:
                dataset_path = os.getcwd()
                # f = open("test.txt",'w')
                # f.write(image)
                # f.write(" 1")
                # f.write('\n')
                # f.write("no.png 0")
                # f.close()
                # if not os.path.exists("cache"):
                #     os.mkdir('cache')
                # import shutil
                # if not os.path.exists(os.path.join("cache", image)):
                #     shutil.copyfile(image, os.path.join("cache", image))
                # shutil.copyfile(image, os.path.join("cache", "no.png"))
                self.cfg.data.test.data_prefix =os.path.join(dataset_path, image)
                # self.cfg.data.test.ann_file = os.path.join(dataset_path,'test.txt')
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
                # checkpoint = None
                try:
                    checkpoint = load_checkpoint(model, checkpoint)
                except FileNotFoundError:
                    info = "Error Code: -102. No such checkpoint file:"+ checkpoint
                    raise Exception(info)
                result = single_gpu_test(model,data_loader )
                # os.remove("test.txt")
                # shutil.rmtree("cache")
                f = open(class_path, "r")
                ff = f.readlines()
                f.close()
                # print("\n",np.argmax(result[0]), ff[np.argmax(result[0])][-1:])
                results = []
                for i in range(len(result)):
                    pred_class = ff[np.argmax(result[i])] if ff[np.argmax(result[i])][-1:] != "\n" else ff[np.argmax(result[i])][:-1]
                    if isinstance(np.argmax(result[i]), np.int64):
                        pred_label = int(np.argmax(result[i]))
                    if isinstance(result[i][np.argmax(result[i])], np.float32):
                        pred_score = float(result[i][np.argmax(result[i])])
                    tmp_result = {
                        'pred_label': pred_label, # np.argmax(result[i]),
                        'pred_score':pred_score, # result[i][np.argmax(result[i])],
                        'pred_class':pred_class,
                    }
                    results.append(tmp_result)
            # model.show_result(image, result, show=show, out_file=os.path.join(save_fold, os.path.split(image)[1]))
            chinese_res = []
            for i in range(len(results)):
                tmp = {
                    '标签':results[i]['pred_label'],
                    '置信度': results[i]['pred_score'],
                    '预测结果':results[i]['pred_class']
                }
                # img.append(tmp)
                chinese_res.append(tmp)
            # print(chinese_res)
            self.chinese_res = chinese_res
            print("\n========= finish inference ==========")

        return results

    def load_dataset(self, path,**kwargs):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: "+ next(iter(kwargs.keys()))
            raise Exception(info)
        self.dataset_path = path
        if not isinstance(path, str):
            info = "Error Code: -201. Dataset file type error, which should be <class 'str'> instead of "+ type(path)+"."
            raise Exception(info)
        if not os.path.exists(path):   # 数据集路径不存在
            info = "Error Code: -101. No such dateset directory: "+ path
            raise Exception(info)
        val_set = os.path.join(path, 'val_set')
        val_txt = os.path.join(path, 'val.txt')
        if os.path.exists(val_set) and os.path.exists(val_txt):
            val_num = 0
            for i in os.listdir(val_set):
                val_num += len(os.listdir(os.path.join(val_set,i)))
            if val_num != len(open(val_txt).readlines()):
                info = "Error Code: -201. Dataset file type error. The number of val set images does not match that in val.txt"
                raise Exception(info)
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
