# Faster R-CNN说明文档

> [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

## 简介

Faster R-CNN 是目标检测领域最为经典双阶段的方法之一，通过 RPN(Region Proposal  Networks) 区域提取网络和 R-CNN 网络联合训练实现高效目标检测。

## 发展历程

1. R-CNN。首先通过传统的 selective search 算法在图片上预取 2000 个左右 Region Proposal；接着将这些 Region Proposal 通过前处理统一尺寸输入到 CNN  中进行特征提取；然后把所提取的特征输入到 SVM 支持向量机中进行分类；最后对分类后的 Region Proposal 进行 bbox  回归。此时算法的整个过程较为繁琐，速度也较慢。
2. Fast R-CNN。首先通过传统的  selective search 算法在图片上预取 2000 个左右 Region Proposal；接着对整张图片进行特征提取；然后利用  Region Proposal 坐标在 CNN 的最后一个特征图上进去 RoI 特征图提取；最后将所有 RoI  特征输入到分类和回归模块中。此时算法的整个过程相比 R-CNN 得到极大的简化，但依然无法联合训练。
3. Faster R-CNN。首先通过可学习的 RPN 网络进行 Region Proposal 的预取；接着利用 Region Proposal 坐标在  CNN 的特征图上进行 RoI 特征图提取；然后利用 RoI Pooling  层进行空间池化使其所有特征图输出尺寸相同；最后将所有特征图输入到后续的 FC 层进行分类和回归。此时算法的整个过程一气呵成，实现了端到端训练。

## 特点：区域候选网络(Region Proposal Networks,RPN) 

Faster R-CNN 的出现改变了整个目标检测算法的发展历程。之所以叫做 two-stage 检测器，原因是其**包括一个区域提取网络  RPN 和 RoI Refine 网络 R-CNN，同时为了将 RPN 提取的不同大小的 RoI 特征图组成 batch 输入到后面的  R-CNN 中，在两者中间还插入了一个 RoI Pooling 层，可以保证任意大小特征图输入都可以变成指定大小输出**。简要结构图如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143881188-ab87720f-5059-4b4e-a928-b540fb8fb84d.png" height="300"/>
</div>



## 特点：特征金字塔(Feature Pyramid Networks,FPN)

Faster R-CNN 之后，考虑到多尺度预测问题，后续又提出了改进版本**特征金字塔 FPN(Feature Pyramid Networks for Object Detection)**。 通过分析目前目标检测中存在的图像金字塔、单层预测和多层预测问题，提出了一个简单的，通过从上到下路径和横向连接，结合高分辨率、弱语义信息的特征层和低分辨率、强语义信息的特征融合，实现类似图像金字塔效果，**顶层特征通过上采样和低层特征做融合，而且每层都是独立预测的**，效果显著，如下图所示：

<div align=center>
<img src="https://pic4.zhimg.com/80/v2-5a78ef8716761b468a1ae5f4d9810d13_720w.jpg" height="300"/>
</div>

由于其强大的性能，更加模块化现代化的设计，现在提到 Faster R-CNN, 一般默认是指的 FPN 网络。

## 优点

- 双阶段网络相比于单阶段网络，性能优越，检测精度高。
- 可以解决多尺度、小目标问题。
- 通用性强，适用各种目标检测任务，且便于迁移。
  

## 适用领域

目标检测

## 预训练模型

R50-FPN:  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth)

## 参考文献

```
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```

