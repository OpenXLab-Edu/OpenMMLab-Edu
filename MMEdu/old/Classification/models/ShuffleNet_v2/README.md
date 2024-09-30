# ShuffleNet_v2说明文档

>[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164.pdf)

## 简介

ShuffleNet_v2是旷视提出的轻量化网络，在移动端应用非常广泛。

ShuffleNet_v2中提出了一个关键点，之前的轻量级网络都是通过计算网络复杂度的一个间接度量，即FLOPs为指导。通过计算浮点运算量来描述轻量级网络的快慢。但是从来不直接考虑运行的速度。在移动设备中的运行速度不仅仅需要考虑FLOPs，还需要考虑其他的因素，比如内存访问成本(memory access cost)和平台特点(platform characterics)。

所以，ShuffleNet_v2直接通过控制不同的环境来直接测试网络在设备上运行速度的快慢，而不是通过FLOPs来判断。其次，所有性能指标都应该直接在目标平台上进行评估以反映真实效果。

## 特点
### 1：当卷积层的输入特征矩阵与输出特征矩阵channel相等时MAC最小(保持FLOPs不变时)

现代网络通常采用depthwise separable convolutions ，其中pointwise convolution(即1×1卷积)占了复杂性的大部分。通过研究1×1卷积的核形，发现其形状由两个参数指定:输入通道c1和输出通道c2的数量。设h和w为feature map的空间大小，1×1卷积的FLOPs为$B = h*w*c_{1}*c_{2}$。内存访问成本(MAC)，即内存访问操作数，为$MAC = hw(c_{1}+c_{2})+c_{1}*c_{2}$
这两个术语分别对应于输入/输出特性映射的内存访问和内核权重。这条公式也可以看成由三个部分组成：第一部分是$hwc_{1}$，对应的是输入特征矩阵的内存消耗；第二部分是$hwc_{2}$，对应的是输出特征矩阵的内存消耗。第三部分是$c_{1}*c_{2}$,对应的是卷积核的内存消耗。由均值不等式$\frac{c_{1}+c_{2}}{2} ≥ \sqrt{c_{1}c_{2}}$可以得出：

$MAC≥2hw\sqrt{c_{1}c_{2}}+c_{1}c_{2}≥2\sqrt{hwB}+\frac{B}{hw}$，其中$B = hwc_{1}c_{2}$。因此，MAC有一个由FLOPs给出的下限。当输入和输出通道的数量相等时，它达到下界。

<div align=center> <img src="https://img-blog.csdnimg.cn/20210624165017779.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70" /> </div>

表中实验通过改变比率c1: c2报告了在固定总FLOPs时的运行速度。可见，当c1: c2接近1:1时，MAC变小，网络评估速度加快。

### 2：当GConv的groups增大时(保持FLOPs不变时)，MAC也会增大

组卷积是现代网络体系结构的核心。它通过将所有通道之间的密集卷积改变为稀疏卷积(仅在通道组内)来降低计算复杂度(FLOPs)。一方面，它允许在一个固定的FLOPs下使用更多的channels，并增加网络容量(从而提高准确性)。然而，另一方面，增加的通道数量导致更多的MAC。

$MAC=hw(c_{1}+c_{2})+\frac{c_{1}c_{2}}{g}=hwc_{1}+\frac{Bg}{c_{1}}+\frac{B}{hw}$，其中g为分组数，$B=\frac{hwc_{1}c_{2}}{g}$为FLOPs。不难看出，给定固定的输入形状$c_{1}*h*w$，计算代价B, MAC随着g的增长而增加

<div align=center> <img src="https://img-blog.csdnimg.cn/20210624171010136.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70" /> </div>

很明显，使用大量的组数会显著降低运行速度。例如，在GPU上使用8group比使用1group(标准密集卷积)慢两倍以上，在ARM上慢30%。这主要是由于MAC的增加。所以使用比较大组去进行组卷积是不明智的。对速度会造成比较大的影响。

### 3：网络设计的碎片化程度越高，速度越慢

虽然这种碎片化结构已经被证明有利于提高准确性，但它可能会降低效率，因为它对GPU等具有强大并行计算能力的设备不友好。它还引入了额外的开销，比如内核启动和同步。

<div align=center> <img src="https://img-blog.csdnimg.cn/20210624171846516.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70" /> </div>

为了量化网络分片如何影响效率，我们评估了一系列不同分片程度的网络块。具体来说,每个构造块由1到4个1 × 1的卷积组成，这些卷积是按顺序或平行排列的。每个块重复堆叠10次。块结构上图所示。

<div align=center> <img src="https://img-blog.csdnimg.cn/20210624171715757.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70" /> </div>

表中实验结果显示，在GPU上碎片化明显降低了速度，如4-fragment结构比1-fragment慢3倍。在ARM上，速度降低相对较小。

一个比较容易理解为啥4-fragment结构比较慢的说法是，4-fragment结构需要等待每个分支处理完之后再进行下一步的操作，也就是需要等待最慢的那一个。所以，效率是比较低的。


### 4：Element-wise操作带来的影响是不可忽视的

<div align=center> <img src="https://img-blog.csdnimg.cn/20210624160632344.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70" /> </div>

轻量级模型中，元素操作占用了相当多的时间，特别是在GPU上。这里的元素操作符包括ReLU、AddTensor、AddBias等。将depthwise convolution作为一个element-wise operator，因为它的MAC/FLOPs比率也很高

<div align=center> <img src="https://img-blog.csdnimg.cn/20210624172518594.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70" /> </div>

可以看见表中报告了不同变体的运行时间并能观察到，在移除ReLU和shortcut后，GPU和ARM都获得了大约20%的加速。

这里主要突出的是，这些操作会比我们想象当中的要耗时。

### 总结：
基于上述准则和实证研究，我们得出结论:一个高效的网络架构应该

- 使用“平衡”卷积(等信道宽度);
- 了解使用群卷积的代价;
- 降低碎片化程度;
- 减少元素操作。

这些理想的属性依赖于平台特性(如内存操作和代码优化)，这些特性超出了理论FLOPs。在实际的网络设计中应该考虑到这些因素。而轻量级神经网络体系结构最新进展大多基于FLOPs的度量，没有考虑上述这些特性。

## ShuffleNetV2网络结构

ShuffleNetV1的结构主要采用了两种技术：pointwise group convolutions与bottleneck-like structures。然后引入“channel shuffle”操作，以实现不同信道组之间的信息通信，提高准确性。

both pointwise group convolutions与bottleneck structures均增加了MAC，与G1和G2不符合。这一成本是不可忽视的，特别是对于轻型机型。此外，使用太多group违反G3。shortcut connection中的元素element-wise add操作也是不可取的，违反了G4。因此，要实现高模型容量和高效率，关键问题是如何在不密集卷积和不过多分组的情况下，保持大量的、同样宽的信道。

其中图c对应stride=1的情况，图d对应stride=2的情况

<div align=center> <img src="https://img-blog.csdnimg.cn/20210624174338622.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70" width="70%" /> </div>

为此，ShuffleNetV2做出了改进，如图( c )所示，在每个单元的开始，c特征通道的输入被分为两个分支（在ShuffleNetV2中这里是对channels均分成两半）。根据G3，不能使用太多的分支，所以其中一个分支不作改变，另外的一个分支由三个卷积组成，它们具有相同的输入和输出通道以满足G1。两个1 × 1卷积不再是组卷积，而改变为普通的1x1卷积操作，这是为了遵循G2（需要考虑组的代价）。卷积后，两个分支被连接起来，而不是相加(G4)。因此，通道的数量保持不变(G1)。然后使用与ShuffleNetV1中相同的“channels shuffle”操作来启用两个分支之间的信息通信。需要注意，ShuffleNet v1中的“Add”操作不再存在。像ReLU和depthwise convolutions 这样的元素操作只存在于一个分支中。

对于空间下采样，单元稍作修改，移除通道分离操作符。因此，输出通道的数量增加了一倍。具体结构见图（d）。所提出的构建块( c )( d )以及由此产生的网络称为ShuffleNet V2。基于上述分析，我们得出结论，该体系结构设计是高效的，因为它遵循了所有的指导原则。积木重复堆叠，构建整个网络。

总体网络结构类似于ShuffleNet v1，如表所示。只有一个区别:在全局平均池之前增加了一个1 × 1的卷积层来混合特性，这在ShuffleNet v1中是没有的。与下图类似，每个block中的通道数量被缩放，生成不同复杂度的网络，标记为0.5x，1x，1.5x，2x

<div align=center> <img src="https://img-blog.csdnimg.cn/20210624175820178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70" width="70%" /> </div>

## 总结 
ShuffleNet v2不仅高效，而且准确。主要有两个原因：

- 每个构建块的高效率使使用更多的特征通道和更大的网络容量成为可能
- 在每个块中，有一半的特征通道直接穿过该块并加入下一个块。这可以看作是一种特性重用，就像DenseNet和CondenseNet的思想一样。

## 参考文献

~~~
@inproceedings{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={116--131},
  year={2018}
}
~~~