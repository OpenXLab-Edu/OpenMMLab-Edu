# Mask R-CNN

> [Mask R-CNN](https://arxiv.org/abs/1703.06870)

## 简介

Mask R-CNN由Faster R-CNN拓展而来，可以同时在一个网络中做目标检测和实例分割，特征提取采用ResNet-FPN的架构，另外多加了一个Mask预测分支，并引入了RoI Align代替Faster RCNN中的RoI Pooling。

<div align=center> <img src="https://pica.zhimg.com/v2-ea09be343cb10c6e740cf8de3d0bccec_1440w.jpg?source=172ae18b" width="60%"/> </div>



## 特点：Mask预测分支

我们知道在Faster R-CNN中，对于每个ROI（文中叫candidate object）主要有两个输出，

- 一个输出是分类结果，也就是预测框的标签；
- 一个输出是回归结果，也就是预测框的坐标offset。

而Mask R-CNN则是添加了第三个输出：object mask，也就说对每个ROI都输出一个mask，该支路是通过FCN网络（如上图中的两个卷积层）来实现的。以上这三个输出支路相互之间都是平行关系，相比其他先分割再分类的实例分割算法相比，这种平行设计不仅简单而且高效。

## 特点：RoIAlign

RoIPool的目的是从RPN(区域候选网络)确定的ROI中导出较小的特征图(eg 7x7)，ROI的大小各不相同，但是RoIPool后都变成了7x7大小。RPN网络会提出若干RoI的坐标以[x,y,w,h]表示，然后输入RoI Pooling，输出7x7大小的特征图供分类和定位使用。

问题就出在RoI Pooling的输出大小是7x7上，如果RON网络输出的RoI大小是8*8的，那么无法保证输入像素和输出像素是一一对应，首先他们包含的信息量不同（有的是1对1，有的是1对2），其次他们的坐标无法和输入对应起来。这对分类没什么影响，但是对分割却影响很大。RoIAlign的输出坐标使用插值算法得到，不再是简单的量化；每个grid中的值也不再使用max，同样使用差值算法。


## 优点

- **精准**：识别精度高，分割准确
- **高效**：掩码层只给整个系统增加一小部分计算量，运行速度很快
- **泛化性强**：使用FasterRCNN的架构，可以兼容泛化到其他任务上

## 适用领域

广泛应用于目标检测、实例分割等领域。

- 在COCO比赛目标检测(object detection)、实例分割(instance segmentation)和行人关键点检测(person keypoint detection)任务上获得第一名

## 预训练模型

|                           Backbone                           | Mem (GB) | box AP | mask AP |                            Config                            |                           Download                           |
| :----------------------------------------------------------: | :------: | :----: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   [R-50-FPN](./mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py)    |   4.1    |  40.9  |  37.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154.log.json) |
|  [R-101-FPN](./mask_rcnn_r101_fpn_mstrain-poly_3x_coco.py)   |   6.1    |  42.7  |  38.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244.log.json) |
| [x101-32x4d-FPN](./mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_coco.py) |   7.3    |  43.6  |  39.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_coco_20210524_201410-abcd7859.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_coco_20210524_201410.log.json) |
| [X-101-32x8d-FPN](./mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco.py) |   10.3   |  44.3  |  39.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco_20210607_161042-8bd2c639.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco_20210607_161042.log.json) |
| [X-101-64x4d-FPN](./mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py) |   10.4   |  44.5  |  39.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447.log.json) |

## 参考文献

```
@article{He_2017,
   title={Mask R-CNN},
   journal={2017 IEEE International Conference on Computer Vision (ICCV)},
   publisher={IEEE},
   author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
   year={2017},
   month={Oct}
}
```

