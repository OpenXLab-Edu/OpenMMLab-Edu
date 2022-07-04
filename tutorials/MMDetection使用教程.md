# MMDetection使用文档

### 简介

MMDetection的主要功能：输出图片或视频中出现的多个对象名称，同时用方框框出对象所在方形区域。

<img src="imgs\image-20220408192155931.png" alt="image-20220408192155931" style="zoom:40%;" />

其支持的SOTA模型有FasterRCNN、Yolov3等。

### 使用说明

MMEdu中预置了“车牌检测”数据集，并且已经预训练了权重（路径：/checkpoint/det_model/plate/latest.pth）。在demo文件夹中，还提供了一张汽车图片（含车牌）用于测试。

#### 1.直接推理（支持CPU）

如果想快速上手体验MMDetection的话，我们建议您使用我们已经预训练好的模型和权重文件进行推理，提供一张图片测试推理的准确度。


执行代码如下:

```python
import base # 测试版需要（发布版不需要）
from MMEdu import MMDetection as det # 导入mmdet模块

img = 'car_plate.png' # 指定进行推理的图片路径，我们使用demo文件夹中提供的图片
model = det(backbone="FasterRCNN") # 实例化MMDetection模型
checkpoint = '../checkpoints/det_model/plate/latest.pth' # 指定使用的模型权重文件
class_path = '../dataset/classes/det_classes.txt' # 指定训练集的路径
result = model.inference(image=img, show=True, class_path=class_path,checkpoint_path = checkpoint) # 在CPU上进行推理
model.print_result() # 输出结果
# 同时您可以修改show的值来决定是否需要显示结果图片，此处默认显示结果图片
```

运行结果如图：

<img src="imgs\image-20220408191108835.png" alt="image-20220408191108835" style="zoom:45%;" />

推理结果图片（带标签的图片）会以原本的文件名称保存在`demo`文件夹下的`det_result`文件夹下，如果在`demo`下没有发现该文件夹，不用担心，系统会自动建立。当然，您可以自己指定保存文件夹的名称。

您也可以将收集的图片放在一个文件夹下，然后指定文件夹路径进行一组图片的批量推理。如在demo文件夹下新建一个det_testIMG文件夹放图片，运行下面这段代码。

~~~python
img = 'det_testIMG/' # 指定进行推理的一组图片的路径
model = det(backbone="FasterRCNN") # 实例化MMDetection模型
checkpoint = '../checkpoints/det_model/plate/latest.pth' # 指定使用的模型权重文件
class_path = '../dataset/classes/det_classes.txt' # 指定训练集的路径
result = model.inference(image=img, show=True, class_path=class_path,checkpoint_path = checkpoint) # 在CPU上进行推理
model.print_result() # 输出结果
# 同时您可以修改show的值来决定是否需要显示结果图片，此处默认显示结果图片
~~~

同时您会发现当前目录下`‘det_result’`文件夹里出现了这组图片的推理结果图，每张图片的结果与您收集的图片同名，您可以到这个文件夹下查看推理结果。

接下来对为您讲述代码规则：

- **图片准备**

```python
img = 'car_plate.png' # 指定推理图片的路径，直接在代码所在的demo文件夹中选择图片
```

如果使用自己的图片的话，只需要修改img的路径即可（绝对路径和相对路径均可）。

- **实例化模型**

```python
model = det(backbone='FasterRCNN') # 实例化MMDetection模型
```

这里对于`MMDetection`模型提供的参数进行解释，`MMDetection`支持传入的参数是`backbone`。

`backbone`：指定使用的`MMDetection`模型，默认使用 `'FasterRCNN'`，当然读者可以自行修改该参数以使用不同模型。

- **模型推理**

```python
model.inference(image=img, show=True, class_path=class_path,checkpoint_path = checkpoint) # 在CPU上进行推理
```

将所需要推理图片的路径传入`inference`函数中即可进行推理，我们这里传入了四个参数，`image`代表的就是推理图片的路径，`show`代表是否需要显示结果图片，`class_path`代表训练集的路径，`checkpoint`代表指定使用的模型权重文件。

- **参数详解**：

在Detection_Edu中对于`inference`函数还有其他的传入参数，在这里进行说明：

`device`：推理所用的设备，默认为`'cpu'`，如果电脑支持GPU，也可以将参数修改为`'cuda:0'`，使用GPU进行推理。

`checkpoint`：指定使用的模型权重文件，默认参数为`None`，如果没有指定模型权重文件，那么我们将会使用默认的模型权重文件进行推理。 

`image`：推理图片的路径。

`show`：布尔值，默认为`True`，表示推理后是否显示推理结果。

`rpn_threshold` & `rcnn_threshold`: 0～1之间的数值。由于FasterRCNN为一个两阶段的检测模型，这两个参数分别表示两个阶段对于检测框的保留程度，高于这个数值的框将会被保留（这里如果同学们设置过低，也可能会发现图中出现了多个框）。

`class_path`：指定训练集的路径，默认参数为`"../dataset/classes/det_classes.txt"`。

`save_fold`：保存的图片名，数据结构为字符串，默认参数为`'det_result'`，用户也可以定义为自己想要的名字。

（最后两个参数的使用，我们将在下一部分进行详细举例解释）。

#### 2.训练模型

使用下面的代码即可简单体验MMDetection的训练过程，我们以车牌的识别为例，为您进行详细的介绍。


在运行代码之前，您需要先拥有一个数据集，这里我们为您提供车牌检测数据集。

数据集文件结构如下：

![image-20220408210420560](imgs\image-20220408210420560.png)

**coco**文件夹中包含两个文件夹`annotations`和`images`，分别存储标注信息以及图片数据，每个文件夹下面有`train`和`valid`两个`json`文件。

- 代码展示

~~~python
model = det(backbone='FasterRCNN') # 实例化模型，不指定参数即使用默认参数。
model.num_classes = 1 # 进行车牌识别，此时只有一个类别。
model.load_dataset(path='../dataset/det/coco') # 从指定数据集路径中加载数据
model.save_fold = '../checkpoints/det_model/plate' # 设置模型的保存路径
model.train(epochs=3, validate=True) # 设定训练的epoch次数以及是否进行评估
~~~


**详细说明**

实例化模型的代码在前面说过就不再赘述。

- **指定类别数量**

~~~python
model.num_classes = 1 # 进行车牌识别，此时只有一个类别
~~~

- **加载数据集**

~~~python
model.load_dataset(path='../dataset/det/coco') # 从指定数据集路径中加载数据
~~~

这个函数只需要传入一个`path`参数即训练数据集的路径，函数的作用是修改模型中关于数据集路径的配置文件，从而确保我们在训练时不会找错文件。

- **指定模型参数存储位置**

~~~python
model.save_fold = '../checkpoints/det_model/plate'
~~~


- **模型训练**

~~~python
model.train(epochs=10, validate=True) # 设定训练的epoch次数以及是否进行评估
~~~

表示训练10个轮次，并在训练结束后用检验集进行评估。


- **参数详解**

`train`函数支持很多参数，为了降低难度，MMEdu已经给绝大多数的参数设置了默认值。根据具体的情况修改参数，可能会得到更好的训练效果。下面来详细说明`train`函数的各个参数。

`random_seed`：随机种子策略，默认为`0`即不使用，使用随机种子策略会减小模型算法结果的随机性。

`save_fold`：模型的保存路径，默认参数为`./checkpoints/det_model/`，如果不想模型保存在该目录下，可自己指定路径。

`distributed`：布尔值，只能为`True`或者`False`，默认参数为`False`，设为`True`时即使用分布式训练。

`epochs`：默认参数为`100`，用于指定训练的轮次，而在上述代码中我们设置为`10`。

`validate`：布尔值，只能为`True`或者`False`，默认参数为`True`，在训练结束后，设定是否需要在校验集上进行评估，`True`则是需要进行评估。

`metric`：验证指标，默认参数为`'bbox'`，在进行模型评估时会计算预测的检测框和实际检测框相交的多少，数值越高说明模型性能越好，我们在运行完程序之后也会看到这个结果。

`save_best`：验证指标，默认参数为`'bbox_mAP'`，在进行模型评估时会计算分类准确率，数值越高说明模型性能越好，运行完程序之后会将这个结果保存。

`optimizer`：进行迭代时的优化器，默认参数为`SGD`，`SGD`会在训练的过程中迭代计算mini-bath的梯度。

`lr`：学习率，默认参数为`1e-3`即`0.001`，指定模型进行梯度下降时的步长。简单解释就是，学习率过小，训练过程会很缓慢，学习率过大时，模型精度会降低。

`weight_decay`：权值衰减参数，用来调节模型复杂度对损失函数的影响，防止过拟合，默认值为`1e-3`即`0.001`。

`checkpoint`: 默认为'None'，表示在训练过程中使用初始化权重。如果使用训练得到的模型（或预训练模型），此参数传入一个模型路径，我们的训练将基于传入的模型参数继续训练。

执行上述代码之后的运行结果如下图

<img src="imgs\image-20220408211213751.png" alt="image-20220408211213751" style="zoom:60%;" />

而在`checkpoints\det_model`文件夹中我们会发现多了两种文件，一个是`None.log.json`文件，它记录了我们模型在训练过程中的一些参数，比如说学习率`lr`，所用时间`time`，以及损失`loss`等；另一个文件是.pth文件，这个是我们在训练过程中所保存的模型。


#### 3.继续训练

在这一步中，我们会教您加载之前训练过的模型接着训练，如果您觉得之前训练的模型epoch数不够的话或者因为一些客观原因而不得不提前结束训练，相信下面的代码会帮到您。

~~~python
model = det(backbone='FasterRCNN') # 初始化实例模型
model.num_classes = 1  # 进行车牌识别，此时只有一个类别。
model.load_dataset(path='../dataset/det/coco') # 配置数据集路径
model.save_fold = '../checkpoints/det_model/plate' # 设置模型的保存路径
checkpoint='../checkpoints/det_model/plate/latest.pth' # 指定使用的模型权重文件
model.train(epochs=3, validate=True, checkpoint=checkpoint) # 进行再训练
~~~

这里我们有一个参数在之前的[训练模型](####2.训练模型)过程中没有提及，那就是`train`函数中的`checkpoint`参数，这个放到这里就比较好理解，它的意思是指定需要进行再训练的模型路径，当然你也可以根据你需要训练的不同模型而调整参数。

#### 4.SOTA模型介绍

目前MMDetection支持的SOTA模型有FaterRCNN、Yolov3等，这些模型的作用和适用场景简介如下。

- **FasterRCNN**

采用双阶检测方法，可以解决多尺度、小目标问题，通用性强。

- **Yolov3**

只进行一次检测，速度较快，适用于稍微大的目标检测问题。