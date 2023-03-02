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
import cv2

warnings.filterwarnings("ignore")


class MMDetection:
    def sota():
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
            "SSD_Lite":os.path.join(self.file_dirname, 'models', 'SSD_Lite/SSD_Lite.py'),
            # "Mask_RCNN":os.path.join(self.file_dirname, 'models', 'Mask_RCNN/Mask_RCNN.py'),
            # 下略
        }
        self.num_classes = num_classes
        self.chinese_res = None
        self.is_sample = False

    def train(self, random_seed=0, save_fold=None, distributed=False, validate=True, device='cpu',
              metric='bbox', save_best='bbox_mAP', optimizer="SGD", epochs=100, lr=0.001, weight_decay=0.001,
              Frozen_stages=1,
              checkpoint=None, batch_size=None):

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
            if "RCNN" not in self.backbone: # 单阶段
                self.cfg.model.bbox_head.num_classes =self.num_classes
            elif self.backbone == "FasterRCNN": # rcnn系列 双阶段
                self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
            elif self.backbone == "Mask_RCNN":
                self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
                self.cfg.model.roi_head.mask_head.num_classes = self.num_classes



        self.load_dataset(self.dataset_path)
        # 添加需要进行检测的类名
        if self.backbone in ["Yolov3"]:
            self.cfg.classes = self.get_classes(self.cfg.data.train.dataset.ann_file)
        else:
            self.cfg.classes = self.get_classes(self.cfg.data.train.ann_file)

        # 分别为训练、测试、验证添加类名
        if self.backbone in ["Yolov3"]:
            self.cfg.data.train.dataset.classes = self.cfg.classes
        else:
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
        if batch_size is not None:
            self.cfg.data.samples_per_gpu = batch_size
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
            sample_result = r"[{'类别标签': 0, '置信度': 1.0, '坐标': {'x': 26, 'y': 81, 'w': 497, 'h': 414}},{'类别标签': 1, '置信度': 1.0, '坐标': {'x1': 250, 'y1': 103, 'x2': 494, 'y2': 341}}]"
            print(sample_result)
        else:
            print("检测结果如下：")
            print(self.chinese_res)
        return self.chinese_res

    def load_checkpoint(self, checkpoint=None, device='cpu',
                        rpn_threshold=0.7, rcnn_threshold=0.7):
        print("========= begin inference ==========")
        if self.num_classes != -1 and self.backbone not in ["Yolov3", "SSD_Lite"]:
            self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes

        if checkpoint:
            # 加载数据集及配置文件的路径
            # self.load_dataset(self.dataset_path)
            # 修正检测的目标
            # self.cfg.classes = self.get_class(class_path)
            self.cfg.classes = torch.load(checkpoint, map_location=torch.device('cpu'))['meta']['CLASSES']
            self.cfg.data.train.classes = self.cfg.classes
            self.cfg.data.test.classes = self.cfg.classes
            self.cfg.data.val.classes = self.cfg.classes
            if "RCNN" not in self.backbone: # 单阶段
                self.cfg.model.bbox_head.num_classes = len(self.cfg.classes)
            else: # rcnn系列 双阶段
                self.cfg.model.roi_head.bbox_head.num_classes =  len(self.cfg.classes)
            # self.cfg.model.roi_head.bbox_head.num_classes = len(self.cfg.classes)
            self.infer_model = init_detector(self.cfg, checkpoint, device=device)
            self.infer_model.CLASSES = self.cfg.classes
        else:
            self.infer_model = init_detector(self.cfg, self.checkpoint, device=device)
        if self.backbone not in ["Yolov3", "SSD_Lite"]: self.infer_model.test_cfg.rpn.nms.iou_threshold = 1 - rpn_threshold
        if self.backbone not in ["Yolov3", "SSD_Lite"]: self.infer_model.test_cfg.rcnn.nms.iou_threshold = 1 - rcnn_threshold

    def fast_inference(self, image, show=False, save_fold='det_result'):
        img_array = mmcv.imread(image)
        try:
            self.infer_model
        except:
            print("请先使用load_checkpoint()方法加载权重！")
            return
        result = inference_detector(self.infer_model, img_array)  # 此处的model和外面的无关,纯局部变量
        self.infer_model.show_result(image, result, show=show,
                                     out_file=os.path.join(save_fold, os.path.split(image)[1]))
        chinese_res = []
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
        # print("========= finish inference ==========")
        return result

    def inference(self, device='cpu',
                  checkpoint=None,
                  image=None,
                  show=True,
                  rpn_threshold=0.7,
                  rcnn_threshold=0.7,
                  save_fold='det_result',
                  ):
        # self.cfg.classes = self.get_class(class_path)
        # self.num_classes = len(self.get_class(class_path))
        # if self.num_classes != -1:
        #     if "RCNN" not in self.backbone: # 单阶段
        #         self.cfg.model.bbox_head.num_classes =self.num_classes
        #     elif self.backbone == "FasterRCNN": # rcnn系列 双阶段
        #         self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
        #     elif self.backbone == "Mask_RCNN":
        #         self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
        #         self.cfg.model.roi_head.mask_head.num_classes = self.num_classes
        
        if image == None:
            self.is_sample = True
            sample_return = """
        [array([[ 26.547777  ,  81.55447   , 497.37015   , 414.4934    ,
          1.0]], dtype=float32), 
          array([[2.5098564e+02, 1.0334784e+02, 4.9422855e+02, 3.4187744e+02,
        1.0],  dtype=float32)]
            """
            return sample_return
        self.is_sample = False
        print("========= begin inference ==========")

        if self.num_classes != -1 and self.backbone not in ["Yolov3", "SSD_Lite"] :
            self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes

        if checkpoint:
            # 加载数据集及配置文件的路径
            # self.load_dataset(self.dataset_path)
            # 修正检测的目标
            self.cfg.classes = torch.load(checkpoint, map_location=torch.device('cpu'))['meta']['CLASSES']

            self.num_classes = len(self.cfg.classes)
            if self.num_classes != -1:
                if "RCNN" not in self.backbone: # 单阶段
                    self.cfg.model.bbox_head.num_classes =self.num_classes
                elif self.backbone == "FasterRCNN": # rcnn系列 双阶段
                    self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
                elif self.backbone == "Mask_RCNN":
                    self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
                    self.cfg.model.roi_head.mask_head.num_classes = self.num_classes
            self.cfg.data.train.classes = self.cfg.classes
            self.cfg.data.test.classes = self.cfg.classes
            self.cfg.data.val.classes = self.cfg.classes
            if self.backbone not in ["Yolov3", "SSD_Lite"]:
                self.cfg.model.roi_head.bbox_head.num_classes = len(self.cfg.classes)
            model = init_detector(self.cfg, checkpoint, device=device)
            model.CLASSES = self.cfg.classes
        else:
            model = init_detector(self.cfg, self.checkpoint, device=device)
        # model = build_detector(self.cfg.model, train_cfg=self.cfg.get(
        #     'train_cfg'), test_cfg=self.cfg.get('test_cfg'))
        # if not checkpoint:
        #     model.init_weights()
        # else:
        #     checkpoint = os.path.abspath(checkpoint)  # pip修改2
        #     load_checkpoint(model, checkpoint, map_location=torch.device(device))

        if self.backbone not in ["Yolov3", "SSD_Lite"]: model.test_cfg.rpn.nms.iou_threshold = 1 - rpn_threshold
        if self.backbone not in ["Yolov3", "SSD_Lite"]: model.test_cfg.rcnn.nms.iou_threshold = 1 - rcnn_threshold

        results = []
        if (image[-1] != '/'):
            img_array = mmcv.imread(image)
            result = inference_detector(
                model, img_array)  # 此处的model和外面的无关,纯局部变量
            if show == True:
                show_result_pyplot(model, image, result)
            model.show_result(image, result, show=show, out_file=os.path.join(save_fold, os.path.split(image)[1]))
            chinese_res = []
            for i in range(len(result)):
                for j in range(result[i].shape[0]):
                    tmp = {}
                    tmp['类别标签'] = i
                    tmp['置信度'] = result[i][j][4]
                    tmp['坐标'] = {"x1": int(result[i][j][0]), "y1": int(
                        result[i][j][1]), 'x2': int(result[i][j][2]), 'y2': int(result[i][j][3])}
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
                result = inference_detector(
                    model, img_dir + img)  # 此处的model和外面的无关,纯局部变量
                model.show_result(img_dir + img, result,
                                  out_file=os.path.join(save_fold, img))
                chinese_res = []
                for i in range(len(result)):
                    for j in range(result[i].shape[0]):
                        tmp = {}
                        tmp['类别标签'] = i
                        tmp['置信度'] = result[i][j][4]
                        tmp['坐标'] = {"x1": int(result[i][j][0]), "y1": int(
                            result[i][j][1]), 'x2': int(result[i][j][2]), 'y2': int(result[i][j][3])}
                        # img.append(tmp)
                        chinese_res.append(tmp)
                chinese_results.append(chinese_res)
                results.append(result)
            self.chinese_res = chinese_results
        print("========= finish inference ==========")
        return results

    def load_dataset(self, path):
        self.dataset_path = path

        # 数据集修正为coco格式
        if self.backbone in ["Yolov3"]:
            self.cfg.data.train.dataset.img_prefix = os.path.join(self.dataset_path, 'images/train/')
            self.cfg.data.train.dataset.ann_file = os.path.join(self.dataset_path, 'annotations/train.json')
        else:
            self.cfg.data.train.img_prefix = os.path.join(self.dataset_path, 'images/train/')
            self.cfg.data.train.ann_file = os.path.join(self.dataset_path, 'annotations/train.json')

        self.cfg.data.val.img_prefix = os.path.join(self.dataset_path, 'images/test/')
        self.cfg.data.val.ann_file = os.path.join(self.dataset_path, 'annotations/valid.json')

        self.cfg.data.test.img_prefix = os.path.join(self.dataset_path, 'images/test/')
        self.cfg.data.test.ann_file = os.path.join(self.dataset_path, 'annotations/valid.json')

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

    def convert(self, checkpoint=None, backend="ONNX", out_file="convert_model.onnx",device='cpu'):
        import os.path as osp
        from mmdet.core.export import build_model_from_cfg

        ashape = self.cfg.test_pipeline[1].img_scale
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
        if self.backbone not in ["Yolov3", "SSD_Lite"] :
            self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
        else:
            self.cfg.model.bbox_head.num_classes = self.num_classes

        # build the model and load checkpoint
        # detector = build_detector(self.cfg.model)
        # model = build_model_from_cfg(self.config, checkpoint)
        
        if checkpoint:
            # 加载数据集及配置文件的路径
            # self.load_dataset(self.dataset_path)
            # 修正检测的目标
            self.cfg.classes = torch.load(checkpoint, map_location=torch.device('cpu'))['meta']['CLASSES']
            self.num_classes = len(self.cfg.classes)
            if self.num_classes != -1:
                if "RCNN" not in self.backbone: # 单阶段
                    self.cfg.model.bbox_head.num_classes =self.num_classes
                elif self.backbone == "FasterRCNN": # rcnn系列 双阶段
                    self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
                elif self.backbone == "Mask_RCNN":
                    self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
                    self.cfg.model.roi_head.mask_head.num_classes = self.num_classes
            self.cfg.data.train.classes = self.cfg.classes
            self.cfg.data.test.classes = self.cfg.classes
            self.cfg.data.val.classes = self.cfg.classes
            if self.backbone not in ["Yolov3", "SSD_Lite"]:
                self.cfg.model.roi_head.bbox_head.num_classes = len(self.cfg.classes)
            model = init_detector(self.cfg, checkpoint, device=device)
            model.CLASSES = self.cfg.classes
        else:
            model = init_detector(self.cfg, self.checkpoint, device=device)
        if self.backbone not in ["Yolov3", "SSD_Lite"]: model.test_cfg.rpn.nms.iou_threshold = 0.3 # 1 - rpn_threshold
        if self.backbone not in ["Yolov3", "SSD_Lite"]: model.test_cfg.rcnn.nms.iou_threshold = 0.3 # 1 - rcnn_threshold


        #detector = build_detector(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        # detector.CLASSES = self.num_classes
        normalize_cfg = parse_normalize_cfg(self.cfg.test_pipeline)
        input_img = osp.join(osp.dirname(__file__), './demo/demo.jpg')
        if backend == "ONNX" or backend == 'onnx':
            pytorch2onnx(
                # detector,
                model,
                input_img,
                input_shape,
                normalize_cfg,
                show=False,
                output_file=out_file,
                verify=False,
                test_img=None,
                do_simplify=False)
        else:
            print("Sorry, we only suport ONNX up to now.")
        with open(out_file.replace(".onnx", ".py"), "w+") as f:
            tp = str(self.cfg.test_pipeline).replace("},","},\n\t")
            # if class_path != None:
            #     classes_list = self.get_class(class_path)
            classes_list = torch.load(checkpoint, map_location=torch.device('cpu'))['meta']['CLASSES']

            gen0 = """
import onnxruntime as rt
import BaseData
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret_flag,image = cap.read()
cap.release()
"""
            gen_sz = """
image = cv2.resize(image,(sz_h,sz_w))
tag = 
"""
            gen1 = """
sess = rt.InferenceSession('
"""
            gen2 = """', None)
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]
dt = BaseData.ImageData(image, backbone="
"""

            gen3 = """")
input_data = dt.to_tensor()
outputs = sess.run(output_names, {input_name: input_data})

boxes = outputs[0]
labels = outputs[1][0]
img_height, img_width = image.shape[:2]
size = min([img_height, img_width]) * 0.001
text_thickness = int(min([img_height, img_width]) * 0.001)

idx = 0
for box in zip(boxes[0]):
    x1, y1, x2, y2, score = box[0]
    label = tag[labels[idx]]
    idx = idx + 1
    caption = f'{label}{int(score * 100)}%'
    if score >= 0.15:
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, caption, (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

cv2.imwrite("result.jpg", image)
""" 
            ashape = self.cfg.test_pipeline[1].img_scale
            # if class_path != None:
            gen = gen0.strip("\n") + '\n' + gen_sz.replace('sz_h',str(ashape[0])).replace('sz_w',str(ashape[1])).strip('\n') + str(classes_list)+ "\n" + gen1.strip("\n") + out_file + gen2.strip("\n") + str(self.backbone) + gen3
            # else:
            #     gen = gen0.strip("tag = \n") + "\n\n" + gen1.strip("\n")+out_file+ gen2.strip("\n") + str(self.backbone) + gen3.replace("tag[labels[idx]]", "labels[idx]")
            f.write(gen)


def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config

def pytorch2onnx(model,
                 input_img,
                 input_shape,
                 normalize_cfg,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 test_img=None,
                 do_simplify=False,
                 dynamic_export=None,
                 skip_postprocess=False):
    from mmdet.core.export import build_model_from_cfg, preprocess_example_input
    from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
    from functools import partial
    from mmcv import Config, DictAction
    import onnx
    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }
    # prepare input
    one_img, one_meta = preprocess_example_input(input_config)
    img_list, img_meta_list = [one_img], [[one_meta]]

    if skip_postprocess:
        warnings.warn('Not all models support export onnx without post '
                      'process, especially two stage detectors!')
        model.forward = model.forward_dummy
        torch.onnx.export(
            model,
            one_img,
            output_file,
            input_names=['input'],
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=show,
            opset_version=opset_version)

        print(f'Successfully exported ONNX model without '
              f'post process: {output_file}')
        return

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        rescale=False)

    output_names = ['dets', 'labels']
    if model.with_mask:
        output_names.append('masks')
    input_name = 'input'
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {
            input_name: {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        if model.with_mask:
            dynamic_axes['masks'] = {0: 'batch', 1: 'num_dets'}

    torch.onnx.export(
        model,
        img_list,
        output_file,
        input_names=[input_name],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes)

    model.forward = origin_forward

    if do_simplify:
        import onnxsim

        from mmdet import digit_version

        min_required_version = '0.4.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnxsim>={min_required_version}'

        model_opt, check_ok = onnxsim.simplify(output_file)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            warnings.warn('Failed to simplify ONNX model.')
    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        #print(model.CLASSES)
        # wrap onnx model
        onnx_model = ONNXRuntimeDetector(output_file, model.CLASSES, 0)
        if dynamic_export:
            # scale up to test dynamic shape
            h, w = [int((_ * 1.5) // 32 * 32) for _ in input_shape[2:]]
            h, w = min(1344, h), min(1344, w)
            input_config['input_shape'] = (1, 3, h, w)

        if test_img is None:
            input_config['input_path'] = input_img

        # prepare input once again
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = [one_img], [[one_meta]]

        # get pytorch output
        with torch.no_grad():
            pytorch_results = model(
                img_list,
                img_metas=img_meta_list,
                return_loss=False,
                rescale=True)[0]

        img_list = [_.cuda().contiguous() for _ in img_list]
        if dynamic_export:
            img_list = img_list + [_.flip(-1).contiguous() for _ in img_list]
            img_meta_list = img_meta_list * 2
        # get onnx output
        onnx_results = onnx_model(
            img_list, img_metas=img_meta_list, return_loss=False)[0]
        # visualize predictions
        score_thr = 0.3
        if show:
            out_file_ort, out_file_pt = None, None
        else:
            out_file_ort, out_file_pt = 'show-ort.png', 'show-pt.png'

        show_img = one_meta['show_img']
        model.show_result(
            show_img,
            pytorch_results,
            score_thr=score_thr,
            show=True,
            win_name='PyTorch',
            out_file=out_file_pt)
        onnx_model.show_result(
            show_img,
            onnx_results,
            score_thr=score_thr,
            show=True,
            win_name='ONNXRuntime',
            out_file=out_file_ort)

        # compare a part of result
        if model.with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res, p_res, rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')
