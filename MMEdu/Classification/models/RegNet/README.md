# RegNet说明文档

>[Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

## 简介

RegNet是何凯明大神团队提出的用NAS得到的分类网络，该网络在轻量级网络领域，低FLOPs的RegNet模型也能达到很好的效果，和MobileNetV2以及ShuffleNetV2性能有的一比

<div align=center> <img src="https://img-blog.csdnimg.cn/20210304175226519.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" width="70%" /> </div>

与当时分类网络的天花板EfficientNet对比，可以看到RegNetY-8.0GF的错误率比EfficientNet-B5更低，且推理速度(infer)快五倍。

<div align=center> <img src="https://img-blog.csdnimg.cn/20210304175838200.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" width="70%" /> </div>

## RegNet网络结构

<div align=center> <img src="https://img-blog.csdnimg.cn/20210304110827789.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" width="70%" /> </div>

图中展示了网络主要由三部分组成，stem、body和head。

- stem就是一个普通的卷积层（默认包含bn以及relu），卷积核大小为3x3，步距为2，卷积核个数为32.
- body就是由4个stage堆叠组成，如图（b）所示。每经过一个stage都会将输入特征矩阵的height和width缩减为原来的一半。而每个stage又是由一系列block堆叠组成，每个stage的第一个block中存在步距为2的组卷积（主分支上）和普通卷积（捷径分支上），剩下的block中的卷积步距都是1，和ResNet类似。
- head就是分类网络中常见的分类器，由一个全局平均池化层和全连接层构成。

## 特点：AnyNet设计

论文作者说，根据他们的经验将block设计为standard residual bottlenecks block with group convolution即带有组卷积的残差结构（和ResNext的block类似），如下图所示，左图为block的stride=1的情况，右图为block的stride=2的情况：

<div align=center> <img src="https://img-blog.csdnimg.cn/20210304113414303.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" /> </div>

由图可知，主分支都是一个1x1的卷积（包括bn和relu）、一个3x3的group卷积（包括bn和relu）、再接一个1x1的卷积（包括bn）。shortcut捷径分支上当stride=1时不做任何处理，当stride=2时通过一个1x1的卷积（包括bn）进行下采样。图中的r代表分辨率简单理解为特征矩阵的高、宽，当步距s等于1时，输入输出的r保持不变，当s等于2时，输出的r为输入的一半。w代表特征矩阵的channel（注意当s=2时，输入的是$w_{i-1}$, 输出的是$w_i$即chennel会发生变化）。g代表group卷积中每个group的group width，b代表bottleneck ratio即输出特征矩阵的channel缩减为输入特征矩阵channel的$\frac{1}{b}.此时就从AnyNet的设计空间缩小到AnyNetX空间了，该空间也称为$AnyNetX_A。此时的设计空间依旧很大，接着论文中说为了获得有效的模型，又加了些限制：$d_i \leq 16$（有16种可能）, $w_i \leq 1024$且取8的整数倍（有128种可能）， $b_i \in \left\{1, 2, 4\right\}$（有3种可能）, $g_i \in \left\{1, 2, 4, 8, 16,32\right\} $（有6种可能），其中$d_i$表示stage中重复block的次数，由于body中由4个stage组成。那么现在还有大约$10^{18}$种模型配置参数（想要在这么大的空间去搜索基本不可能）：$(16⋅128⋅3⋅6)^{4}≈10^{18}$

接着作者又尝试将所有stage中的block的$b_i$都设置为同一个参数b（shared bottleneck ratio），此时的设计空间记为$AnyNetX_B$，然后在$AnyNetX_A$和$AnyNetX_B$中通过log-uniform sampling采样方法分别采样500的模型，并在imagenet上训练10个epochs，绘制的error-cumulative prob.对比如下图所示：

<div align=center> <img src="https://img-blog.csdnimg.cn/20210305111234929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" /> </div>

通过上图可以发现，将所有stage中的block的$b_i$都设置为同一个参数b（shared bottleneck ratio）后并没有什么明显的变化。

剩余详细设计思路可以在[CSDN博客](https://blog.csdn.net/qq_37541097/article/details/114362044)中学习。

## 参考文献

~~~
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
~~~