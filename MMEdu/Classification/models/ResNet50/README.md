# ResNet说明文档

> [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

## 简介

深度残差网络（Deep residual network, ResNet）的提出是计算机视觉领域的一件里程碑事件。

众所周知，网络越深获得的特征越丰富，然而由于梯度消失或梯度爆炸等原因，过深的网络反而会导致性能变差，而ResNet有效的解决了这个问题，它通过学习残差并且加上一个恒等映射来拟合期望的潜在映射。

## 特点：残差结构

ResNet中经典的残差模块如下图所示

<div align=center> <img src="https://user-images.githubusercontent.com/26739999/142574068-60cfdeea-c4ec-4c49-abb2-5dc2facafc3b.png" width="40%"/> </div>

常见算法由$x$直接映射$y$，而ResNet将此过程分为两部分：一是恒等映射(identity mapping)，即直接映射完全相等的$x$，如上图中右侧曲线部分；二是残差映射(residual mapping)，残差的定义是预测值(y)和观测值(x)之间的距离，即 $F(x):=y-x$，也是网络主要训练的权重参数，如上图中左侧直线部分。因此最终输出的是 $y=F(x)+x$。

对于“随着网络加深，性能下降”的问题，ResNet提供了两种选择方式，也就是恒等映射和残差映射，如果网络已经到达最优，继续加深网络，残差映射会趋向于0，只剩下恒等映射，这样理论上网络一直处于最优状态了，网络的性能也就不会随着深度增加而降低了。

学习残差比学习原始特征更为容易，就好比找出两幅画的区别要比根据一幅画画出另一幅容易的多。

## ResNet网络结构

ResNet网络是参考了VGG19网络，在其基础上加入了残差结构，如下图所示：

<div align=center> <img src="https://pic2.zhimg.com/80/v2-7cb9c03871ab1faa7ca23199ac403bd9_720w.jpg" width="60%"/> </div>

可以看到，ResNet相比普通网络每两层间增加了恒等映射（右侧弧形箭头）。且ResNet网络层数相比VGG有明显提升，更多网络层意味着更多参数，也意味着更好的拟合能力。

以上是34层的ResNet（简称ResNet34）的结构图，还可以构建ResNet50、101、152等更深的网络。如此之深的网络在ResNet之前完全不可行，因为过于庞大的模型意味着计算资源的大量占用、效率低下，还面临梯度消失、梯度爆炸导致的性能退化问题。至今为止，ResNet仍是难以替代的主流模型之一。

## 优点

- 解决了网络随深度增加而性能退化的问题

- 结构简单，效果拔群

## 适用领域

广泛应用于分类，检测，分割等领域。

- 在ImageNet比赛分类(classification)、定位(localization)任务上获得第一名
- 在COCO比赛检测(detection)、分割(segmentation)任务上获得第一名
- Alpha zero也使用了ResNet

## 参考文献

```
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

