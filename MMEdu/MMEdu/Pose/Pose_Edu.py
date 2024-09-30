# import mmcv
# from mmcv import Config
# import mmpose
# from mmpose.apis import (inference_top_down_pose_model, init_pose_model,train_model,
#                          vis_pose_result, process_mmdet_results)
# from mmdet.apis import inference_detector, init_detector
# from mmpose.datasets import build_dataset
# from mmpose.models import build_posenet
# import cv2
# import os
# import tempfile
# import os.path as osp
# from tqdm import tqdm
# import sys
# import urllib

# class MMPose:
#     def __init__(self, 
#         backbone_det='FasterRCNNpose',
#         backbone = 'HrNet32', 
#         #note if inference need HrNet (64)
#         dataset_path = None
#         ):
#         self.backbone_ckpt_dict={
#             "SCNet":"https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192-6920f829_20200709.pth",
#             "FasterRCNNpose":"https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
#             "RTMPose":"https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/triangle_dataset/ckpt/0512/rtmpose_s_triangle_300-34bfaeb2_20230512.pth",

#         }
#         # 获取外部运行py的绝对路径
#         self.cwd = os.path.dirname(os.getcwd())
#         # 获取当前文件的绝对路径
#         self.save_fold = None
#         self.file_dirname = os.path.dirname(os.path.abspath(__file__))

#         self.backbone_det = backbone_det
#         backbone_det_path = os.path.join(self.file_dirname, 'models', self.backbone_det)
#         ckpt_cfg_list = list(os.listdir(backbone_det_path))
#         det_flag = False
#         for item in ckpt_cfg_list:
#             # if item[-1] == 'y'and item[0]!='_':
#             if item == str(self.backbone_det)+".py":
#                 self.det_config = os.path.join(backbone_det_path, item)
#             # elif item[-1] == 'h':
#             elif item == str(self.backbone_det)+".pth":
#                 det_flag = True
#                 self.det_checkpoint = os.path.join(backbone_det_path, item)
#             else:
#                 # print("Warning!!! There is an unrecognized file in the backbone folder.")
#                 pass
#         if det_flag == False:
#             self.det_checkpoint = os.path.join(backbone_det_path, str(self.backbone_det)+'.pth')
#             self.download(self.backbone_ckpt_dict[self.backbone_det],self.det_checkpoint)


#         self.backbone = backbone
#         backbone_path = os.path.join(self.file_dirname, 'models', self.backbone)
#         ckpt_cfg_list = list(os.listdir(backbone_path))
#         pose_flag = False
#         for item in ckpt_cfg_list:
#             print(item)
#             # if item[-1] == 'y' and item[0]!='_':
#             if item == str(self.backbone)+".py":
#                 self.pose_config = os.path.join(backbone_path, item)
#             # elif item[-1] == 'h':
#             elif item == str(self.backbone)+".pth":
#                 pose_flag = True
#                 self.pose_checkpoint = os.path.join(backbone_path, item)
#         if pose_flag == False:
#             self.pose_checkpoint = os.path.join(backbone_path, str(self.backbone)+'.pth')
#             self.download(self.backbone_ckpt_dict[self.backbone],self.pose_checkpoint)
    
#         self.cfg_det = Config.fromfile(self.det_config)
#         self.cfg = Config.fromfile(self.pose_config)
        
#         self.dataset_path = dataset_path

#         return None


#     def train(self, random_seed=0, save_fold=None, checkpoint = None, distributed=False, validate=True,
#               metric='PCK', save_best = 'PCK',optimizer="Adam", epochs=100, lr=5e-4,
#               resume_from = None,
#               eval_interval = 10,
#               log_interval = 5,
#               ):
#         print("========= begin training ==========")
#         # 如果外部不指定save_fold
#         if not self.save_fold:
#             # 如果外部也没有传入save_fold，我们使用默认路径
#             if not save_fold:
#                 self.save_fold = os.path.join(self.cwd, 'checkpoints/pose_model')
#             # 如果外部传入save_fold，我们使用传入值
#             else:
#                 self.save_fold = save_fold

#         # self.cfg = Config.fromfile(self.backbonedict[self.backbone])
#         # print(self.cfg.pretty_text)
#         self.cfg.gpu_ids = range(1)
#         self.cfg.work_dir = self.save_fold
#         self.cfg.load_from = checkpoint
#         self.cfg.resume_from = resume_from
#         self.cfg.seed = random_seed

#         self.cfg.evaluation.interval = eval_interval
#         self.cfg.evaluation.metric = metric  # 验证指标
#         self.cfg.evaluation.save_best = save_best  # 验证指标
    

#         # self.cfg.model.backbone.frozen_stages = Frozen_stages
#         # set log interval
#         self.cfg.log_config.interval = log_interval
#         self.cfg.total_epochs = epochs  # 最大的训练轮次
#         self.cfg.optimizer.lr = lr  # 学习率
#         self.cfg.optimizer.type = optimizer  # 优化器

#         datasets = [build_dataset(self.cfg.data.train)]

#         # build model
#         model = build_posenet(self.cfg.model)

#         # create work_dir
#         mmcv.mkdir_or_exist(self.cfg.work_dir)

#         # train model
#         train_model(
#             model, datasets, self.cfg, distributed=distributed, validate=validate, meta=dict())
#         print("========= finish training ==========")
#         return None

#     def _inference(self,det_model,pose_model,img,work_dir,name,show,i):
#         mmdet_results = inference_detector(det_model, img)
#         person_results = process_mmdet_results(mmdet_results, cat_id=1)
#         pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
#                                                                 img,
#                                                                 person_results,
#                                                                 bbox_thr=0.3,
#                                                                 format='xyxy',
#                                                                 dataset=pose_model.cfg.data.test.type)
#         vis_result = vis_pose_result(pose_model,
#                                 img,
#                                 pose_results,
#                                 dataset=pose_model.cfg.data.test.type,
#                                 show=show)
#         with tempfile.TemporaryDirectory() as tmpdir:
#             if not os.path.exists(work_dir):   ##目录存在，返回为真
#                 os.makedirs(work_dir) 

#             file_name = osp.join(work_dir, name+str(i)+'.png')
#             cv2.imwrite(file_name, vis_result)
#         return pose_results

#     def inference(self,
#                   device='cuda:0',
#                   is_trained=False,
#                   pretrain_model='./checkpoints/pose_model/latest.pth',
#                   img=None,
#                   show=False,
#                   work_dir='./img_result/',
#                   name='pose_result'):
#         """
#         params:
#             device: 推理设备,可选参数: ('cuda:int','cpu')
#             is_trained: 是否使用本地预训练的其他模型进行训练
#             pretrain_model: 如果使用其他模型，则传入模型路径
#             img: 推理图片的路径或文件夹名
#             show: 是否对推理结果进行显示
#             work_dir: 推理结果图片的保存文件夹
#             name: 推理结果保存的名字
#         return:
#             pose_results: 推理的结果数据，一个列表，其中包含若干个字典，每个字典存储对应检测的人体数据。
#         """
#         if not pretrain_model:
#             pretrain_model = os.path.join(self.cwd, 'checkpoints/pose_model/latest.pth')
#         print("========= begin inference ==========")

#         if is_trained == True:
#             self.pose_checkpoint = pretrain_model

#         # initialize pose model
#         pose_model = init_pose_model(self.pose_config, self.pose_checkpoint,device = device)
#         # initialize detector
#         print(self.det_config)
#         det_model = init_detector(self.det_config, self.det_checkpoint,device=device)


#         # inference img
#         if img[-1:] != '/':
#             pose_results = self._inference(det_model,pose_model,img,work_dir,name,show,0)
#             print('Image result is save as %s.png' % (name))

#         else:
#         # inference directory
#             img_dir = img
#             print("inference for directory: %s \n" % (img_dir))
#             for i,img in enumerate(tqdm(os.listdir(img_dir))):
#                 pose_results = self._inference(det_model,pose_model,img_dir+img,work_dir,name,show,i)
#             print('Finish! Image result is save in %s \n' % (work_dir))
#         return pose_results

#     def load_dataset(self, path):

#         self.dataset_path = path

#         #数据集修正为 images train.json val.json 形式
#         # cfg.data_root = 'data/coco_tiny'
#         self.cfg.data.train.type = 'PoseDataset'
        
#         self.cfg.data.train.ann_file = os.path.join(self.dataset_path, 'train.json')
#         self.cfg.data.train.img_prefix = os.path.join(self.dataset_path, 'images/')

#         self.cfg.data.val.type = 'PoseDataset'
#         self.cfg.data.val.ann_file = os.path.join(self.dataset_path, 'val.json')
#         self.cfg.data.val.img_prefix = os.path.join(self.dataset_path, 'images/')

#         self.cfg.data.test.type = 'PoseDataset'
#         self.cfg.data.test.ann_file = os.path.join(self.dataset_path, 'val.json')
#         self.cfg.data.test.img_prefix = os.path.join(self.dataset_path, 'images/')
    
#     def download(self,url,save_path):
#         def  _progress(block_num,block_size,total_size):
#             sys.stdout.write('\r >> Downloading %s %.1f%%'%(url, float(block_num * block_size)/ float(total_size) *100.0))
#             sys.stdout.flush()
#         urllib.request.urlretrieve(url,save_path,_progress)

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union

import cv2
import json
import numpy as np
import warnings
if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
import onnxruntime as ort
import time
import cv2

class MMPose:
    def __init__(self, task="body17"):
        self.task_dict = {
            "body17":"rtmpose-m-80e511.onnx",
            "body26":"...",
            "wholebody133":"...",
            "face106":"...",
            "hand21":"..."
        }
        self.task = task
    
    def inference(self,device='cpu', checkpoint=None,data=None, save_fold='det_result',show=False):
        if checkpoint is None:
            checkpoint = self.task_dict[self.task]
        # model_path='rtmpose-ort/rtmpose-s-0b29a8.onnx'
        model = ort.InferenceSession(checkpoint, None)
        h, w = model.get_inputs()[0].shape[2:]
        model_input_size = (w, h)
        # image_path='rtmpose-ort/000000147979.jpg'
        if isinstance(data,str):
            img = cv2.imread(data)
        else:
            img = data
        # 前处理
        # start_time = time.time()

        resized_img, center, scale = mmpose_preprocess(img, model_input_size)
        input_tensor = [resized_img.transpose(2, 0, 1)]
        input_name = model.get_inputs()[0].name
        output_names = [o.name for o in model.get_outputs()]
        # end_time = time.time()
        # print('前处理耗时：',end_time-start_time)
        # 模型推理
        # start_time = time.time()
        outputs = model.run(output_names, {input_name: input_tensor})
        print("outputs",len(outputs),outputs[0].shape,outputs[1].shape)
        # end_time = time.time()
        # print('推理耗时：',end_time-start_time)
        # 后处理
        # start_time = time.time()
        keypoints, scores = mmpose_postprocess(outputs, model_input_size, center, scale)
        # end_time = time.time()
        # print('后处理耗时：',end_time-start_time)
        # print('推理结果：')
        # print(keypoints)
        if show:
            # 绘制查看效果
            if self.task == 'hand':
                sketch = {
                    'red':[[0,1],[1,2],[2,3],[3,4]],
                    'orange':[[0,5],[5,6],[6,7],[7,8]],
                    'yellow':[[0,9],[9,10],[10,11],[11,12]],
                    'green':[[0,13],[13,14],[14,15],[15,16]],
                    'blue':[[0,17],[17,18],[18,19],[19,20]]
                }
            elif self.task =='body26':
                print("aaaaaaaaaaaaaa")
                sketch = {
                   'red':[[0,1],[1,2],[2,0],[2,4],[1,3],[0,17],[0,18]],
                    'orange':[[18,6],[8,6],[10,8]],
                    'yellow':[[18,19],[19,12],[19,11]],
                    'green':[[12,14],[14,16],[16,23],[21,16],[25,16]],
                    'blue':[[11,13],[13,15],[15,20],[15,22],[15,24]],
                    'purple':[[18,5],[5,7],[7,9]],
                }
            import matplotlib.pyplot as plt
            plt.imshow(plt.imread(data))
            for j in range(keypoints.shape[0]):
                for i in range(keypoints.shape[1]):
                    plt.scatter(keypoints[j][i][0],keypoints[j][i][1],c='b',s=10)
            for color in sketch.keys():
                # print(color,sketch[color])
                for [fx,fy] in sketch[color]:
                    plt.plot([keypoints[0][fx][0],keypoints[0][fy][0]],[keypoints[0][fx][1],keypoints[0][fy][1]],color=color)
            plt.show()
        return keypoints

def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale

def value2list(value: Any, valid_type: Union[Type, Tuple[Type, ...]],
               expand_dim: int) -> List[Any]:
    """If the type of ``value`` is ``valid_type``, convert the value to list
    and expand to ``expand_dim``.

    Args:
        value (Any): value.
        valid_type (Union[Type, Tuple[Type, ...]): valid type.
        expand_dim (int): expand dim.

    Returns:
        List[Any]: value.
    """
    if isinstance(value, valid_type):
        value = [value] * expand_dim
    return value


def check_type(name: str, value: Any,
               valid_type: Union[Type, Tuple[Type, ...]]) -> None:
    """Check whether the type of value is in ``valid_type``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_type (Type, Tuple[Type, ...]): expected type.
    """
    if not isinstance(value, valid_type):
        raise TypeError(f'`{name}` should be {valid_type} '
                        f' but got {type(value)}')


def check_length(name: str, value: Any, valid_length: int) -> None:
    """If type of the ``value`` is list, check whether its length is equal with
    or greater than ``valid_length``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_length (int): expected length.
    """
    if isinstance(value, list):
        if len(value) < valid_length:
            raise AssertionError(
                f'The length of {name} must equal with or '
                f'greater than {valid_length}, but got {len(value)}')


def check_type_and_length(name: str, value: Any,
                          valid_type: Union[Type, Tuple[Type, ...]],
                          valid_length: int) -> None:
    """Check whether the type of value is in ``valid_type``. If type of the
    ``value`` is list, check whether its length is equal with or greater than
    ``valid_length``.

    Args:
        value (Any): value.
        legal_type (Type, Tuple[Type, ...]): legal type.
        valid_length (int): expected length.

    Returns:
        List[Any]: value.
    """
    check_type(name, value, valid_type)
    check_length(name, value, valid_length)


def color_val_matplotlib(
    colors: Union[str, tuple, List[Union[str, tuple]]]
) -> Union[str, tuple, List[Union[str, tuple]]]:
    """Convert various input in RGB order to normalized RGB matplotlib color
    tuples,
    Args:
        colors (Union[str, tuple, List[Union[str, tuple]]]): Color inputs
    Returns:
        Union[str, tuple, List[Union[str, tuple]]]: A tuple of 3 normalized
        floats indicating RGB channels.
    """
    if isinstance(colors, str):
        return colors
    elif isinstance(colors, tuple):
        assert len(colors) == 3
        for channel in colors:
            assert 0 <= channel <= 255
        colors = [channel / 255 for channel in colors]
        return tuple(colors)
    elif isinstance(colors, list):
        colors = [
            color_val_matplotlib(color)  # type:ignore
            for color in colors
        ]
        return colors
    else:
        raise TypeError(f'Invalid type for color: {type(colors)}')


def color_str2rgb(color: str) -> tuple:
    """Convert Matplotlib str color to an RGB color which range is 0 to 255,
    silently dropping the alpha channel.

    Args:
        color (str): Matplotlib color.

    Returns:
        tuple: RGB color.
    """
    import matplotlib
    rgb_color: tuple = matplotlib.colors.to_rgb(color)
    rgb_color = tuple(int(c * 255) for c in rgb_color)
    return rgb_color


def convert_overlay_heatmap(feat_map: np.ndarray,
                            img: Optional[np.ndarray] = None,
                            alpha: float = 0.5) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    norm_img = np.zeros(feat_map.shape)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img


def wait_continue(figure, timeout: float = 0, continue_key: str = ' ') -> int:
    """Show the image and wait for the user's input.

    This implementation refers to
    https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

    Args:
        timeout (float): If positive, continue after ``timeout`` seconds.
            Defaults to 0.
        continue_key (str): The key for users to continue. Defaults to
            the space key.

    Returns:
        int: If zero, means time out or the user pressed ``continue_key``,
            and if one, means the user closed the show figure.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent
    is_inline = 'inline' in plt.get_backend()
    if is_inline:
        # If use inline backend, interactive input and timeout is no use.
        return 0

    if figure.canvas.manager:  # type: ignore
        # Ensure that the figure is shown
        figure.show()  # type: ignore

    while True:

        # Connect the events to the handler function call.
        event = None

        def handler(ev):
            # Set external event variable
            nonlocal event
            # Qt backend may fire two events at the same time,
            # use a condition to avoid missing close event.
            event = ev if not isinstance(event, CloseEvent) else event
            figure.canvas.stop_event_loop()

        cids = [
            figure.canvas.mpl_connect(name, handler)  # type: ignore
            for name in ('key_press_event', 'close_event')
        ]

        try:
            figure.canvas.start_event_loop(timeout)  # type: ignore
        finally:  # Run even on exception like ctrl-c.
            # Disconnect the callbacks.
            for cid in cids:
                figure.canvas.mpl_disconnect(cid)  # type: ignore

        if isinstance(event, CloseEvent):
            return 1  # Quit for close.
        elif event is None or event.key == continue_key:
            return 0  # Quit for continue.


def img_from_canvas(canvas: 'FigureCanvasAgg') -> np.ndarray:
    """Get RGB image from ``FigureCanvasAgg``.

    Args:
        canvas (FigureCanvasAgg): The canvas to get image.

    Returns:
        np.ndarray: the output of image in RGB.
    """
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype('uint8')

def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict: The result dict contains two items, "train" and "val", for
        the training log and validate log.

    Example:
        An example output:

        .. code-block:: python

            {
                'train': [
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 100},
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 200},
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 300},
                    ...
                ]
                'val': [
                    {"accuracy/top1": 32.1, "step": 1},
                    {"accuracy/top1": 50.2, "step": 2},
                    {"accuracy/top1": 60.3, "step": 2},
                    ...
                ]
            }
    """
    log_dict = dict(train=[], val=[])
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # A hack trick to determine whether the line is training log.
            mode = 'train' if 'lr' in log else 'val'
            log_dict[mode].append(log)

    return log_dict

def _fix_aspect_ratio(bbox_scale: np.ndarray,
                      aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio.

    Args:
        scale (np.ndarray): The image scale (w, h) in shape (2, )
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.ndarray: The reshaped image scale in (2, )
    """
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat

def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict,
                    img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (dict): The bbox scale of the img.
        bbox_center (dict): The bbox center of the img.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: img after affine transform.
        - np.ndarray[float32]: bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale

def mmpose_preprocess(
    img: np.ndarray, input_size: Tuple[int, int] = (192, 256)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Do preprocessing for RTMPose model inference.

    Args:
        img (np.ndarray): Input image in shape.
        input_size (tuple): Input image size in shape (w, h).

    Returns:
        tuple:
        - resized_img (np.ndarray): Preprocessed image.
        - center (np.ndarray): Center of image.
        - scale (np.ndarray): Scale of image.
    """
    # get shape of image
    img_shape = img.shape[:2]
    bbox = np.array([0, 0, img_shape[1], img_shape[0]])

    # get center and scale
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)

    # do affine transformation
    resized_img, scale = top_down_affine(input_size, scale, center, img)

    # normalize image
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    resized_img = (resized_img - mean) / std

    return resized_img, center, scale

def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals

def mmpose_decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian.

    Args:
        simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
        simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
        simcc_split_ratio (int): The split ratio of simcc.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
        - np.ndarray[float32]: scores in shape (K,) or (n, K)
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores

def mmpose_postprocess(outputs: List[np.ndarray],
                model_input_size: Tuple[int, int],
                center: Tuple[int, int],
                scale: Tuple[int, int],
                simcc_split_ratio: float = 2.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Postprocess for RTMPose model output.

    Args:
        outputs (np.ndarray): Output of RTMPose model.
        model_input_size (tuple): RTMPose model Input image size.
        center (tuple): Center of bbox in shape (x, y).
        scale (tuple): Scale of bbox in shape (w, h).
        simcc_split_ratio (float): Split ratio of simcc.

    Returns:
        tuple:
        - keypoints (np.ndarray): Rescaled keypoints.
        - scores (np.ndarray): Model predict scores.
    """
    # use simcc to decode
    simcc_x, simcc_y = outputs
    keypoints, scores = mmpose_decode(simcc_x, simcc_y, simcc_split_ratio)

    # rescale keypoints
    keypoints = keypoints / model_input_size * scale + center - scale / 2

    return keypoints, scores