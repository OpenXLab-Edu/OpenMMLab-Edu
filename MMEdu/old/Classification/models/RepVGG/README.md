# RepVGG说明文档

>[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)

## 简介

RepVGG是一个分类网路，该网络是在VGG网络的基础上进行改进，主要的改进点包括：（1）在VGG网络的Block块中加入了Identity和残差分支，相当于把ResNet网络中的精华应用 到VGG网络中；（2）模型推理阶段，通过Op融合策略将所有的网络层都转换为Conv3*3，便于模型的部署与加速。 该论文中的包含的亮点包括：（1）网络训练和网络推理阶段使用不同的网络架构，训练阶段更关注精度，推理阶段更关注速度

## ResNeXt网络结构

<div align=center> <img src="https://img-blog.csdnimg.cn/20210116094114197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1daWjE4MTkxMTcxNjYx,size_16,color_FFFFFF,t_70#pic_center" width="70%" /> </div>

上图展示了部分RepVGG网络，图A表示的是原始的ResNet网络，该网络中包含着Conv1*1的残差结构和Identity的残差结构，正是这些残差结构的存在解决了深层网路中的梯度消失问题，使得网络更加易于收敛。图B表示的是训练阶段的RepVGG网络架构，整个网络的主体结构和ResNet网络类似，两个网络中都包含残差结构。两个网络中的主要差异如下所述：（1）RepVGG网络中的残差块并没有跨层，如图中的绿框所示；（2）整个网络包含2种残差结构，如图中的绿框和红框所示，绿框中的残差结构仅仅包含Conv1*1残差分支；红框中不仅包含Conv1*1的残差结构，而且包含Identity残差结构。由于残差结构具有多个分支，就相当于给网络增加了多条梯度流动的路径，训练一个这样的网络，其实类似于训练了多个网络，并将多个网络融合在一个网络中，类似于模型集成的思路，不过这种思路更加简单和高效！！！（3）模型的初始阶段使用了简单的残差结构，随着模型的加深，使用了复杂的残差结构，这样不仅仅能够在网络的深层获得更鲁邦的特征表示，而且可以更好的处理网络深层的梯度消失问题。图C表示的是推理阶段的RepVGG网络，该网络的结构非常简单，整个网络均是由Conv3*3+Relu堆叠而成，易于模型的推理和加速。
  这种架构的主要优势包括：（1）当前大多数推理引擎都对Conv3*3做了特定的加速，假如整个网络中的每一个Conv3*3都能节省3ms，如果一个网络中包含30个卷积层，那么整个网络就可以节省3*30=90ms的时间，这还是初略的估算。（2）当推理阶段使用的网络层类别比较少时，我们愿意花费一些时间来完成这些模块的加速，因为这个工作的通用性很强，不失为一种较好的模型加速方案。（3）对于残差节点而言，需要当所有的残差分支都计算出对应的结果之后，才能获得最终的结果，这些残差分支的中间结果都会保存在设备的内存中，这样会对推理设备的内存具有较大的要求，来回的内存操作会降低整个网络的推理速度。而推理阶段首先在线下将模型转换为单分支结构，在设备推理阶段就能更好的提升设备的内存利用率，从而提升模型的推理速度，更直观的理解请看下图。总而言之，模型推理阶段的网络结构越简单越能起到模型加速的效果。

<div align=center> <img src="https://img-blog.csdnimg.cn/20210116100939886.png#pic_center" width="70%" /> </div>

## 特点：重参数化

<div align=center> <img src="https://img-blog.csdnimg.cn/20210116102119451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1daWjE4MTkxMTcxNjYx,size_16,color_FFFFFF,t_70#pic_center" /> </div>

上图展示了模型推理阶段的重参数化过程，其实就是一个OP融合和OP替换的过程。图A从结构化的角度展示了整个重参数化流程， 图B从模型参数的角度展示了整个重参数化流程。整个重参数化步骤如下所示：

步骤1-首先通过式3将残差块中的卷积层和BN层进行融合，该操作在很多深度学习框架的推理阶段都会执行。图中的灰色框中执行Conv3*3+BN层的融合，图中的黑色矩形框中执行Conv1*1+BN层的融合，图中的黄色矩形框中执行Conv3*3(卷积核设置为全1)+BN层的融合。其中Wi表示转换前的卷积层参数，$\mu_{i}$表示BN层的均值，$\sigma_{i}$表示BN层的方差，$\gamma_{i}$和$\beta_{i}$分别表示BN层的尺度因子和偏移因子，$W^{’}$和$b^{’}$分别表示融合之后的卷积的权重和偏置。

步骤2-将融合后的卷积层转换为Conv3\*3，即将具体不同卷积核的卷积均转换为具有3\*3大小的卷积核的卷积。由于整个残差块中可能包含Conv1\*1分支和Identity两种分支，如图中的黑框和黄框所示。对于Conv1\*1分支而言，整个转换过程就是利用3\*3的卷积核替换1\*1的卷积核，具体的细节如图中的紫框所示，即将1\*1卷积核中的数值移动到3\*3卷积核的中心点即可；对于Identity分支而言，该分支并没有改变输入的特征映射的数值，那么我们可以设置一个3\*3的卷积核，将所有的9个位置处的权重值都设置为1，那么它与输入的特征映射相乘之后，保持了原来的数值，具体的细节如图中的褐色框所示。
步骤3-合并残差分支中的Conv3\*3。即将所有分支的权重W和偏置B叠加起来，从而获得一个融合之后的Conv3\*3网络层。

## 总结

RepVGG是一个分类网路，该网络是在VGG网络的基础上进行改进，结合了VGG网络和ResNet网络的思路，主要的创新点包括：（1）在VGG网络的Block块中加入了Identity和残差分支，相当于把ResNet网络中的精华应用 到VGG网络中；（2）模型推理阶段，通过Op融合策略将所有的网络层都转换为Conv3*3，便于模型的部署与加速。（3）这是一个通用的提升性能的Tricks,可以利用该Backbone来替换不同任务中的基准Backbone，算法性能会得到一定程度的提升。
  尽管RepVGG网络具有以上的优势，但是该网络中也存在一些问题，具体的问题包括：（1）从训练阶段转推理阶段之前，需要执行模型重参数化操作；（2）在模型重参数化过程中会增加一些额外的计算量；（3）由于每个残差块中引入了多个残差分支，网络的参数量也增加了一些。

## 参考文献

~~~
@inproceedings{ding2021repvgg,
  title={Repvgg: Making vgg-style convnets great again},
  author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13733--13742},
  year={2021}
}
~~~