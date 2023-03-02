import os
import mmcv
import time
import torch
import numpy as np
import cv2
import PIL
from mmcv import Config
from mmcls.apis import inference_model, init_model, show_result_pyplot, train_model, set_random_seed, single_gpu_test
from mmcls.models import build_classifier
from mmcls.datasets import  build_dataloader,build_dataset
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose

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
        self.image_type = ""

    def train(self, random_seed=0, save_fold=None, distributed=False, validate=True, device="cpu",
              metric='accuracy', save_best='auto', optimizer="SGD", epochs=100, lr=0.01, weight_decay=0.001,
              checkpoint=None,batch_size=None,**kwargs):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        if device not in ['cpu','cuda']:
            info = "Error Code: -301. No such argument: "+ device
            raise Exception(info)
        is_cuda  = torch.cuda.is_available()
        if device == 'cpu' and is_cuda:
            print("You can use  'device=cuda' to accelerate !")
        elif device == 'cuda' and not is_cuda:
            raise Exception("Error Code: -301. Your device doesn't support cuda.")
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
        if  'val_set' not in os.listdir(self.dataset_path):
            print("Unable to validate during training due to lack of validation set !")
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
        if batch_size is not None:
            self.cfg.data.samples_per_gpu = batch_size

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
            print("示例分类结果如下：")
            sample_result = r"[{'标签': 2, '置信度': 1.0, '预测结果': 'scissors'}]"
            print(sample_result)
        else:
            print("分类结果如下：")
            print(self.chinese_res)
        return self.chinese_res


    def load_checkpoint(self,
                  checkpoint=None,
                  device='cpu',
                  **kwargs,
                  ):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: "+ next(iter(kwargs.keys()))
            raise Exception(info)
        if device not in ['cpu','cuda']:
            info = "Error Code: -301. No such argument: "+ device
            raise Exception(info)
        is_cuda  = torch.cuda.is_available()
        if device == 'cpu' and is_cuda:
            print("You can use  'device=cuda' to accelerate !")
        elif device == 'cuda' and not is_cuda:
            raise Exception("Error Code: -301. Your device doesn't support cuda.")
        if checkpoint != None and checkpoint.split(".")[-1] != 'pth':
            info = "Error Code: -202. Checkpoint file type error:"+ checkpoint
            raise Exception(info)
        # if not checkpoint:
            # checkpoint = os.path.join(self.cwd, 'checkpoints/cls_model/hand_gray/latest.pth')
        self.device = device
        classed_name = torch.load(checkpoint)['meta']['CLASSES']
        # classed_name = self.get_class(class_path)
        # self.class_path = class_path
        self.num_classes = len(classed_name)
        if self.num_classes != -1:
            if 'num_classes' in self.cfg.model.backbone.keys():
                self.cfg.model.backbone.num_classes = self.num_classes
            else:
                self.cfg.model.head.num_classes = self.num_classes

        checkpoint = os.path.abspath(checkpoint) # pip修改2
        self.checkpoint = checkpoint
        try:
            self.infer_model = init_model(self.cfg, checkpoint, device=device)
        except FileNotFoundError:
            info = "Error Code: -102. No such checkpoint file:"+ checkpoint
            raise Exception(info)
        self.infer_model.CLASSES = classed_name

    def inference(self, device='cpu',
        checkpoint=None,
        image=None,
        show=True,
        save_fold='cls_result',
        **kwargs,
        ):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        if device not in ['cpu','cuda']:
            info = "Error Code: -301. No such argument: "+ device
            raise Exception(info)
        is_cuda  = torch.cuda.is_available()
        if device == 'cpu' and is_cuda:
            print("You can use  'device=cuda' to accelerate !")
        elif device == 'cuda' and not is_cuda:
            raise Exception("Error Code: -301. Your device doesn't support cuda.")
        if type(image)!=np.ndarray and image == None: # 传入图片为空，示例输出
            self.is_sample = True
            sample_return = """
{'pred_label': 2, 'pred_score': 0.9930743, 'pred_class': 'scissors'}
            """
            return sample_return
        self.is_sample = False
        # if not isinstance(image,(str, np.array)):
        # if not isinstance(image,str): # 传入图片格式，仅支持str图片路径
        #     info = "Error Code: -304. No such argument:"+ image+"which is" +type(image)
        #     raise Exception(info)
        if type(image) != PIL.PngImagePlugin.PngImageFile and type(image) != np.ndarray and not os.path.exists(image):
            info = "Error Code: -103. No such file:"+ image
            raise Exception(info)
        if type(image) != PIL.PngImagePlugin.PngImageFile and os.path.isfile(image) and image.split(".")[-1].lower() not in ["png","jpg","jpeg","bmp"]:
            info = "Error Code: -203. File type error:"+ image
            raise Exception(info)

        if checkpoint != None and checkpoint.split(".")[-1] != 'pth':
            info = "Error Code: -202. Checkpoint file type error:"+ checkpoint
            raise Exception(info)
        if not checkpoint:
            checkpoint = os.path.join(self.cwd, 'checkpoints/cls_model/hand_gray/latest.pth')

        checkpoint = os.path.abspath(checkpoint) # pip修改2
        self.load_checkpoint(device= device, checkpoint=os.path.abspath(checkpoint))
        return self.fast_inference(image=image, show=show,save_fold=save_fold, **kwargs)

    def fast_inference(self, image, show=False, save_fold='cls_result',**kwargs):
        if len(kwargs) != 0:
            info = "Error Code: -501. No such parameter: " + next(iter(kwargs.keys()))
            raise Exception(info)
        # img_array = mmcv.imread(image, flag='color')
        try:
            self.infer_model
        except:
            print("请先使用load_checkpoint()方法加载权重！")
            return


        print("========= begin inference ==========")
        classed_name = self.infer_model.CLASSES
        self.num_classes = len(classed_name)

        results = []
        dataset_path = os.getcwd()
        if type(image) == PIL.PngImagePlugin.PngImageFile: # 以PIL读入图片
            self.image_type = "pil"
            image = np.array(image)

        if type(image) == np.ndarray:
            if self.cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
                self.cfg.data.test.pipeline.pop(0)

            if self.image_type != "pil": self.image_type = "numpy"
            print("{} image".format(self.image_type))

            if self.backbone != "LeNet":  # 单张图片 其他网络
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_array = mmcv.imread(image, flag='color')
                result = inference_model(self.infer_model, img_array)  # 此处的model和外面的无关,纯局部变量
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                data = dict(img=img_gray)
                test_pipeline = Compose(self.cfg.data.test.pipeline)
                data = test_pipeline(data)
                # data = scatter([data], [self.device])[0]
                data_loader = build_dataloader(
                    [data,data],
                    samples_per_gpu=self.cfg.data.samples_per_gpu,
                    workers_per_gpu=self.cfg.data.workers_per_gpu,
                    shuffle=False,
                    round_up=True)
                result = self.batch_infer(self.infer_model, data_loader)
                ff = classed_name
                pred_class = ff[np.argmax(result[0])] if ff[np.argmax(result[0])][-1:] != "\n" else ff[np.argmax(
                    result[0])][:-1]
                result = {
                    'pred_label': np.argmax(result[0]),
                    'pred_score': result[0][np.argmax(result[0])],
                    'pred_class': pred_class,
                }

            self.infer_model.show_result(image, result, show=show, out_file=os.path.join(save_fold, "{}img.jpg".format(self.image_type)))
            chinese_res = []
            tmp = {}
            if isinstance(result['pred_label'], np.int64):
                result['pred_label'] = int(result['pred_label'])
            if isinstance(result['pred_score'], np.float32):
                result['pred_score'] = float(result['pred_score'])
            tmp['标签'] = result['pred_label']
            tmp['置信度'] = result['pred_score']
            tmp['预测结果'] = result['pred_class']
            chinese_res.append(tmp)
            self.chinese_res = chinese_res

        elif os.path.isfile(image): # 以路径读入图片
            if self.backbone != "LeNet": # 单张图片 其他网络
                img_array = mmcv.imread(image, flag='color')
                result = inference_model(self.infer_model, img_array)  # 此处的model和外面的无关,纯局部变量
            else: # 单张图片 Lenet
                imagename = image.split("/")[-1]
                # build the dataloader
                f = open("test.txt", 'w')
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
                self.cfg.data.test.data_prefix = os.path.join(dataset_path, 'cache')
                self.cfg.data.test.ann_file = os.path.join(dataset_path, 'test.txt')
                # self.cfg.data.test.classes = os.path.abspath(self.class_path)

                dataset = build_dataset(self.cfg.data.test)
                # the extra round_up data will be removed during gpu/cpu collect
                data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=self.cfg.data.samples_per_gpu,
                    workers_per_gpu=self.cfg.data.workers_per_gpu,
                    shuffle=False,
                    round_up=True)
                result = self.batch_infer(self.infer_model, data_loader)
                os.remove("test.txt")
                shutil.rmtree("cache")
                ff = classed_name
                pred_class = ff[np.argmax(result[0])] if ff[np.argmax(result[0])][-1:] != "\n" else ff[np.argmax(
                    result[0])][:-1]
                result = {
                    'pred_label': np.argmax(result[0]),
                    'pred_score': result[0][np.argmax(result[0])],
                    'pred_class': pred_class,
                }

            self.infer_model.show_result(image, result, show=show, out_file=os.path.join(save_fold, os.path.split(image)[1]))
            chinese_res = []
            tmp = {}
            if isinstance(result['pred_label'], np.int64):
                result['pred_label'] = int(result['pred_label'])
            if isinstance(result['pred_score'], np.float32):
                result['pred_score'] = float(result['pred_score'])
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
            if self.backbone != "LeNet": # 文件夹 其他网络
                f = open("test.txt", 'w')
                for image_name in os.listdir(image):
                    f.write(image_name)
                    f.write(" 1")
                    f.write('\n')
                f.close()
                self.cfg.data.test.data_prefix = image
                self.cfg.data.test.ann_file = os.path.join(dataset_path, 'test.txt')
                # self.cfg.data.test.classes = os.path.abspath(self.class_path)

                dataset = build_dataset(self.cfg.data.test)
                os.remove("test.txt")
                # the extra round_up data will be removed during gpu/cpu collect
                data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=self.cfg.data.samples_per_gpu,
                    workers_per_gpu=self.cfg.data.workers_per_gpu,
                    shuffle=False,
                    round_up=True)

            else: # 文件夹 Lenet
                dirname = [x.strip() for x in image.split('/') if x.strip() != ''][-1]
                import shutil
                if os.path.exists(os.path.join(dataset_path, 'cache')):
                    shutil.rmtree("cache")
                os.mkdir(os.path.join(dataset_path, 'cache'))
                shutil.copytree(image, os.path.join(dataset_path, 'cache', dirname))
                for i in range(len(classed_name) - 1):
                    dummy_folder = os.path.join(dataset_path, 'cache', 'dummy' + str(i))
                    os.mkdir(dummy_folder)
                self.cfg.data.test.data_prefix = os.path.join(dataset_path, 'cache')
                # self.cfg.data.test.classes = os.path.abspath(self.class_path)
                dataset = build_dataset(self.cfg.data.test)
                # the extra round_up data will be removed during gpu/cpu collect
                data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=self.cfg.data.samples_per_gpu,
                    workers_per_gpu=self.cfg.data.workers_per_gpu,
                    shuffle=False,
                    round_up=True)

            results_tmp = self.batch_infer(self.infer_model, data_loader)

            if os.path.exists(os.path.join(dataset_path, 'cache')):
                shutil.rmtree("cache")

            results = []
            for i in range(len(results_tmp)):
                pred_class = classed_name[np.argmax(results_tmp[i])] if classed_name[np.argmax(results_tmp[i])][-1:] != "\n" else classed_name[ np.argmax(results_tmp[i])][:-1]
                if isinstance(np.argmax(results_tmp[i]), np.int64):
                    pred_label = int(np.argmax(results_tmp[i]))
                if isinstance(results_tmp[i][np.argmax(results_tmp[i])], np.float32):
                    pred_score = float(results_tmp[i][np.argmax(results_tmp[i])])
                    tmp_result = {
                    'pred_label': pred_label,  # np.argmax(result[i]),
                    'pred_score': pred_score,  # result[i][np.argmax(result[i])],
                    'pred_class': pred_class,
                    }
                    results.append(tmp_result)

            for i, img in enumerate(os.listdir(image)):
                self.infer_model.show_result(os.path.join(image,img), results[i], out_file=os.path.join(save_fold, os.path.split(img)[1]))

            # model.show_result(image, result, show=show, out_file=os.path.join(save_fold, os.path.split(image)[1]))
            chinese_res = []
            for i in range(len(results)):
                tmp = {
                    '标签': results[i]['pred_label'],
                    '置信度': results[i]['pred_score'],
                    '预测结果': results[i]['pred_class']
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
        tot = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if "txt" not in file:
                    impath = os.path.join(root, file)
                    # print("impath", impath)
                    if  "jpg"  in impath or "png" in impath:
                        # print(impath)
                        img = cv2.imread(impath)
                        try:
                            img.shape
                        except AttributeError:
                            info = "Error Code: -201. The image file {} is damaged.".format(impath)
                            raise Exception(info)

                        tot += 1
        val_permit = True
        if 'val.txt' not in os.listdir(path) and 'val_set' in os.listdir(path):
            # print("生成val.txt")
            val_permit, valtxt_path = self.generate_txt(path, "val")
        # test_permit = True
        # if 'test.txt' not in os.listdir(path) and 'test_set' in os.listdir(path):
        #     print("生成test.txt")
        #     test_permit, testtxt_path = self.generate_txt(path, "test")
        class_permit = True
        if 'classes.txt' not in os.listdir(path):
            # print("生成classes.txt")
            training_set = os.path.join(path, 'training_set')
            content = sorted(os.listdir(training_set))
            content = [i+"\n" for i in content]
            try:
                classestxt = open(os.path.join(path,"classes.txt"), mode='w')
            except:
                class_permit = False
                dataset_txt = "dataset_txt"
                if not os.path.exists(dataset_txt):
                    os.mkdir(dataset_txt)
                classtxt_path = os.path.join(dataset_txt,"classes.txt")
                classestxt = open(classtxt_path, mode='w')

            classestxt.writelines(content)
            classestxt.close()

        self.cfg.data.train.data_prefix = os.path.join(self.dataset_path, 'training_set')
        # self.cfg.data.train.classes = os.path.join(self.dataset_path, 'classes.txt')

        self.cfg.data.val.data_prefix = os.path.join(self.dataset_path, 'val_set')
        if val_permit:
            self.cfg.data.val.ann_file = os.path.join(self.dataset_path, 'val.txt')
        else:
            self.cfg.data.val.ann_file = valtxt_path
        # self.cfg.data.val.classes = os.path.join(self.dataset_path, 'classes.txt')

        self.cfg.data.test.data_prefix = os.path.join(self.dataset_path, 'test_set')
        # if test_permit:
        self.cfg.data.test.ann_file = os.path.join(self.dataset_path, 'test.txt')
        # else:
        #     self.cfg.data.test.ann_file = testtxt_path
        # self.cfg.data.test.classes = os.path.join(self.dataset_path, 'classes.txt')
        if class_permit:
            self.cfg.data.train.classes = os.path.join(self.dataset_path, 'classes.txt')
            self.cfg.data.val.classes = os.path.join(self.dataset_path, 'classes.txt')
            self.cfg.data.test.classes = os.path.join(self.dataset_path, 'classes.txt')
        else:
            self.cfg.data.train.classes = classtxt_path
            self.cfg.data.val.classes = classtxt_path
            self.cfg.data.test.classes = classtxt_path

    def generate_txt(self, path, type):
        permit = True
        val_set = os.path.join(path, type+'_set')
        txt_path = os.path.join(path,type+".txt")
        try: 
            valtxt = open(txt_path, mode='w')
        except:
            permit = False
            dataset_txt = "dataset_txt"
            if not os.path.exists(dataset_txt):
                os.mkdir(dataset_txt)
            txt_path = os.path.join(dataset_txt,type+".txt")
            valtxt = open(txt_path, mode='w')

        content = []
        for label, i in enumerate(sorted(os.listdir(val_set))):
            for j in sorted(os.listdir(os.path.join(val_set,i))):
                # print(os.path.join(i,j), label)
                content.append("{} {}\n".format(os.path.join(i,j), label))
        valtxt.writelines(content)
        valtxt.close()
        return permit, txt_path

    def get_class(self, class_path):
        classes = []
        with open(class_path, 'r') as f:
            for name in f:
                classes.append(name.strip('\n'))
        return classes

    def batch_infer(self, model, data_loader):
        results_tmp = []
        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(task_num=len(dataset),start=False)
        from mmcv.utils.timer import Timer
        prog_bar.file.flush()
        prog_bar.timer = Timer()
        for i, data in enumerate(data_loader):
            # data = data.to(device)
            if self.device == "cuda": data = scatter(data, [self.device])[0]
            with torch.no_grad():
                result = model(return_loss=False, **data)

            batch_size = len(result)
            results_tmp.extend(result)

            batch_size = data['img'].size(0)
            for _ in range(batch_size):
                prog_bar.update()
        return results_tmp
    
    def convert(self, checkpoint=None, backend="ONNX", out_file="convert_model.onnx"):
        if not (backend == "ONNX" or backend == 'onnx'):
            print("Sorry, we only suport ONNX up to now.")
            return 
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        classes_list = state_dict['meta']['CLASSES']
        self.num_classes = len(classes_list)
        if self.backbone == 'LeNet':
            from mmcls.models.backbones import LeNet5
            from collections import OrderedDict
            model = LeNet5(num_classes=self.num_classes)
            
            class LeNet5_SoftMax(LeNet5):
                def forward(self, x):
                    x = self.features(x)
                    if self.num_classes > 0:
                        x = self.classifier(x.squeeze())
                        x = torch.softmax(x, dim=0)
                    return (x, )

            model = LeNet5_SoftMax(num_classes=self.num_classes)    
            new_state_dict = OrderedDict()
            for key in state_dict['state_dict']:
                new_state_dict[key[9:]] = state_dict['state_dict'][key]
            model.load_state_dict(new_state_dict)
            dummy_input = torch.randn(1, 1, 32, 32)
            try:
                torch.onnx.export(model, dummy_input, out_file)
                print(f'Successfully exported ONNX model: {out_file}')
            except:
                print('Please use the checkpoint train by MMEdu')
        else:
            ashape = [224,224]
            if len(ashape) == 1:
                input_shape = (1, 3, ashape[0], ashape[0])
            elif len(ashape) == 2:
                input_shape = (
                    1,
                    3,
                ) + tuple(ashape)
            else:
                raise ValueError('invalid input shape')
            self.cfg.model.pretrained = None

            self.cfg.model.head.num_classes = self.num_classes
            # build the model and load checkpoint
            classifier = build_classifier(self.cfg.model)

            if checkpoint:
                load_checkpoint(classifier, checkpoint, map_location='cpu')
            else:
                load_checkpoint(classifier, self.checkpoint, map_location='cpu')

            pytorch2onnx(
                classifier, # 模型，此处是分类器
                input_shape, 
                output_file=out_file,
                do_simplify = False,
                verify =False)

        with open(out_file.replace(".onnx", ".py"), "w+") as f:

            gen0 = """
import onnxruntime as rt
import BaseData
import numpy as np
import cv2

tag = 
"""
            gen1 = """
sess = rt.InferenceSession('
"""
            gen2 = """', None)
input_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

cap = cv2.VideoCapture(0)
ret_flag,Vshow = cap.read()
dt = BaseData.ImageData(Vshow, backbone="
"""

            gen3 = """")
input_data = dt.to_tensor()

pred_onx = sess.run([out_name], {input_name: input_data})
ort_output = pred_onx[0]
idx = np.argmax(ort_output, axis=1)[0]
print('result:' + tag[idx])
""" 
            # if class_path != None:
            gen = gen0.strip("\n") + str(classes_list)+ "\n" + gen1.strip("\n")+out_file+ gen2.strip("\n") + str(self.backbone) + gen3
            # else:
            #     gen = gen0.strip("tag = \n") + "\n\n" + gen1.strip("\n")+out_file+ gen2.strip("\n") + str(self.backbone) + gen3.replace("tag[idx]", "idx")
            f.write(gen)
    
# 模型部署
def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.
    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs

def pytorch2onnx(model,
                input_shape,
                opset_version=11,
                dynamic_export=False,
                show=False,
                output_file='tmp.onnx',
                do_simplify=False,
                verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.
    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    from functools import partial
    import onnxruntime as rt
    from mmcv.onnx import register_extra_symbolics
    model.cpu().eval()

    if hasattr(model.head, 'num_classes'):
        num_classes = model.head.num_classes
    # Some backbones use `num_classes=-1` to disable top classifier.
    elif getattr(model.backbone, 'num_classes', -1) > 0:
        num_classes = model.backbone.num_classes
    else:
        raise AttributeError('Cannot find "num_classes" in both head and '
                            'backbone, please check the config file.')
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas={}, return_loss=False)
    register_extra_symbolics(opset_version)

    # support dynamic shape export
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'probs': {
                0: 'batch'
            }
        }
    else:
        dynamic_axes = {}

    with torch.no_grad():
        torch.onnx.export(
            model, (img_list, ),
            output_file,
            input_names=['input'],
            output_names=['probs'],
            export_params=True,
            keep_initializers_as_inputs=True,
            dynamic_axes=dynamic_axes,
            verbose=show,
            opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')
    model.forward = origin_forward

    if do_simplify:
        import onnx
        import onnxsim
        from mmcv import digit_version

        min_required_version = '0.4.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnxsim>={min_required_version}'

        model_opt, check_ok = onnxsim.simplify(output_file)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            print('Failed to simplify ONNX model.')
    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # test the dynamic model
        if dynamic_export:
            dynamic_test_inputs = _demo_mm_inputs(
                (input_shape[0], input_shape[1], input_shape[2] * 2,
                input_shape[3] * 2), model.head.num_classes)
            imgs = dynamic_test_inputs.pop('imgs')
            img_list = [img[None, :] for img in imgs]

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_list, img_metas={}, return_loss=False)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: img_list[0].detach().numpy()})[0]
        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
