# MMEdu_det使用文档
### MMEdu_det简介

MMEdu_det为MMEdu团队在基于OpenMMlab项目的MMDetection模块研发的教育版工具，在实现MMDetection原始功能的基础上，代码更加精简，适用于教学场景、敏捷开发、迁移开发和新手入门。


主要功能：输出图片或视频中出现的多个对象名称，同时用方框框出对象所在方形区域。

![jupyter](img/det.png)

### Detection_Edu安装流程
首先确保您的电脑相关环境配置符合要求：
```shell
python==3.9.7 
numpy==1.22.2 
torch==1.8.1+cu101 # cpu环境只需保证torch==1.8.1
torchvision==0.9.1 
torchaudio==0.8.1
mmcv-full==1.4.5 
```



#### 安装mmdet及其相关依赖库

- 相关库及其版本如下，其中**ipython**是使用Pose_Edu进行推理显示结果时所需要的模块
```txt
mmdet==2.21.0
ipython
```

- 安装方法1

<font color=#A52A2A size=2 >(这里安装可能后期挪到另一个文件直接进行所有模块的安装操作)</font>

```shell
pip install  {module_name}=={version} -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 例如:
# pip install numpy==1.22.2 mmcv==1.4.5 mmdet==2.21.0 mmcls==0.20.1 mmgen ipython -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 这里将所有的模块和对应的版本号复制到一行中使用清华源进行安装
```

- 安装方法2

在当前目录下新建一个文本文档，命名为`requirements.txt`(如果没有设置显示文件扩展名的话命名为`requirements`即可)。

将上面模块及版本信息复制进`requirements.txt`文件中并保存，执行如下命令。

```shell
pip install -r requirements -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

好了，到目前为止，mmdet的环境已经全部配置完成了，接下来就可以开始模型的推理及训练了。


### Detection_Edu使用说明

#### 1.直接推理（支持CPU）

<font color=#A52A2A size=2 >（这部分可能要根据普适性进行修改）</font>

如果想快速上手体验Detection_Edu的话，我们建议您准备<font color=#A52A2A size=2 >XXXXX（图像）</font>或者使用我们自带的图片去进行单独的推理。


在运行下列代码之前，请访问[Faster_RCNN模型下载地址](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)进行模型的下载并分别放置于如下文件夹中。

**Faster_RCNN**: 
```
OPENMMLAB-EDU
├─MMEdu
   ├─models
        ├─FasterRCNN
            ├─FasterRCNN.pth
            ├─FasterRCNN.py
        ├─ ...
```


执行代码如下:

```python
from MMEdu.Detection_Edu import MMDetection #导入MMDetection模块

img = '../demo/001.jpg' #图片路径 # 指定进行推理的图片路径，我们使用demo文件夹中提供的图片
model = MMDetection(backbone='FasterRCNN') # 实例化MMDetection模型
model.inference(infer_data=img) # 在CPU上进行推理
```

运行结果如图：
接下来对为您讲述代码规则

- **图片准备**：

```python
img = '../demo/001.jpg' # 指定推理图片的路径，这里使用相对路径，先../返回上级目录  然后进入上级目录中的demo文件夹中选择图片
```

如果使用自己的图片的话，只需要修改img的路径即可（绝对路径和相对路径均可）

- **实例化模型**：

```python
model = MMDetection(backbone='FasterRCNN') # 实例化MMDetection模型
```

这里对于`MMDetection`模型提供的参数进行解释，`MMDetection`支持传入两个参数：分别是`backbone`和`dataset_path`。


`backbone`：指定使用的MMDetection模型，默认使用 `'FasterRCNN'`。

`dataset_path`：指定训练集的路径，默认参数是`None`，因为我们现在是进行推理，不用在训练集上进行训练，所以暂不指定，后面用到时会详细介绍。



- **模型推理**：

```python
model.inference(img=img,device='cpu') # 在cpu上进行推理
```
<font color=#A52A2A size=2 >（CPU待添加）</font>

将所需要推理图片的路径传入`inference`函数中即可进行推理，我们这里传入了两个参数，`img`代表的就是推理图片的路径，`device`是推理所用的设备，默认是参数是`'cuda:0'`，使用GPU，我们这里将其赋值为`'cpu'`，给设备不支持GPU的同学做一个参考。

在Pose_Edu中对于`inference`函数还有其他的传入参数，在这里进行说明：

`rpn_threshold` & `rcnn_threshold`: 0～1之间的数值。由于FasterRCNN为一个两阶段的检测模型，这两个参数分别表示两个阶段对于检测框的保留程度，高于这个数值的框将会被保留（这里如果同学们设置过低，也可能会发现图中出现了多个框）。

`show`：布尔值，默认为`True`，表示推理后是否显示推理结果。

`save`：布尔值，默认为`True`，表示对于推理的结果图片是否保存到本地。<font color=#A52A2A size=2 >（具体修改）</font>

`name`：保存的图片名，数据结构为字符串，默认参数为`'det_result'`，用户也可以定义为自己想要的名字。 <font color=#A52A2A size=2 >（具体修改）</font>

`is_trained`: 是否使用本地预训练的其他模型进行训练（True/False），我们会默认加载FasterRCNN上在COCO数据集的模型权重，如果同学们进行了自己数据集的训练，请大家将该参数设为False，并在下一个参数传入模型路径。

`pretrain_model`：默认为`None`， 如果需要从自己的模型进行推理，请大家传入自己的模型路径。

（最后两个参数的使用，我们将在下一部分进行详细举例解释）

我们将进行车牌的检测，这里我展示一下推理后的结果：
![jupyter](img/det1.png)
可以看到，识别到的车牌被绿色的标注框框出，我们注意到这里有一个0.7的数字，表示模型对于此标注框的置信度。

- **结果分析：**

我们这里对于模型推理得到的数据进行查看并且分析，所以用`result`变量进行接受。

~~~python
result = model.inference(img=img,device='cpu') # 在CPU上进行推理并获取结果
print(result)  # 将结果输出进行分析
~~~

<font color=#A52A2A size=2 >result的输出根据到时候图进行修改</font>
```
array([[272.03644, 381.30264, 560.1219 , 472.0304, 0.6982837]],dtype=float32)
```

我们看到输出了一个array格式的5维数据，前四个分别表示检测框的左上角、右下角坐标，最后一个数字为置信度（在图中四舍五入显示为0.7）。



#### 2.训练模型

使用下面的代码即可简单体验Detection_Edu的训练过程，我们以车牌的识别为例，为您进行详细的介绍。


在运行代码之前，您需要下载[Faster_RCNN模型下载地址](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)进行模型的下载并分别放置于如下文件夹中。

**Faster_RCNN**: 
```
OPENMMLAB-EDU
├─MMEdu
   ├─models
        ├─FasterRCNN
            ├─FasterRCNN.pth
            ├─FasterRCNN.py
        ├─ ...
```


对于数据集，
<font color=#A52A2A size=2 >
数据集详细解释。</font>

- 代码展示

~~~python
DATASET_PATH =  XXX# 指定训练数据集的路径
model = MMDetection() # 实例化模型，不指定参数即使用默认参数。
model.num_classes = 1 # 进行车牌识别，此时只有一个类别。
model.load_dataset(path=DATASET_PATH) # 从指定数据集路径中加载数据
model.train(epochs=10, validate=True) # 设定训练的epoch次数以及是否进行评估
~~~


**详细说明**

实例化模型的代码在前面说过就不再赘述。

- 加载数据集：

~~~python
model.load_dataset(path=DATASET_PATH) # 从指定数据集路径中加载数据
~~~

这个函数只需要传入一个`path`参数即训练数据集的路径，函数的作用是修改模型中关于数据集路径的配置文件，从而确保我们在训练时不会找错文件。

这里补充一点：

~~~python
model = MMdetection() # 实例化模型，不指定参数即使用默认参数。
model.load_dataset(path=DATASET_PATH) # 从指定数据集路径中加载数据
~~~


上面两行代码和`model = MMdetection(dataset_path=DATASET_PATH)`这一行代码所实现的结果是一样的。

而我们设置不同的函数和接口实现这个功能是为了确保适用于不同的场景，比如说`model = MMdetection(dataset_path=DATASET_PATH)`这行代码只能在模型的初始化阶段指定数据集路径，而`model.load_dataset(path=DATASET_PATH)`可以在模型初始化之后的任何阶段进行修改，比如我可以先用我们提供的数据集进行一个阶段的训练，然后更换为其他数据集进行下一轮训练。


- 模型训练

~~~python
model.train(epochs=10, validate=True) # 设定训练的epoch次数以及是否进行评估
~~~

其实这一部分的参数比较多，我先解释一些好理解的以及常用的参数。

`epochs`：默认参数为`100`，用于指定训练的轮次，而在上述代码中我们设置为`10`。

`validate`：布尔值，只能为`True`或者`False`，默认参数为`True`，在训练结束后，设定是否需要在校验集上进行评估，`True`则是需要进行评估。

`lr`：学习率，默认参数为`5e-4`即`0.0005`，指定模型进行梯度下降时的步长。简单解释就是，学习率过小，训练过程会很缓慢，学习率过大时，模型精度会降低。

`save_fold`：模型的保存路径，默认参数为`./checkpoints/det_model/`，如果不想模型保存在该目录下，可自己指定路径。

`optimizer`：进行迭代时的优化器，默认参数为`Adam`，`Adam`会在训练的过程中动态调整学习率，避免学习率过小或过大。

`distributed`：布尔值，只能为`True`或者`False`，默认参数为`False`，设为`True`时即使用分布式训练。

`metric`：验证指标，默认参数为`'bbox'`，在进行模型评估时会计算预测的检测框和实际检测框相交的多少，数值越高说明模型性能越好，我们在运行完程序之后也会看到这个结果。

`random_seed`：随机种子策略，默认为`0`即不使用，使用随机种子策略会减小模型算法结果的随机性。

`checkpoint`: 默认为'None'，表示在训练过程中使用初始化权重。如果使用训练得到的模型（或预训练模型），此参数传入一个模型路径，我们的训练将基于传入的模型参数继续训练。



执行上述代码之后的运行结果如下图

<font color=#A52A2A size=2 >
结果图。</font>



#### 3.使用自己训练的模型进行推理

这一步像是把推理和训练结合到了一起，首先**确保你进行了推理和训练的准备工作**，并且使用训练的代码在本地保存了一个模型。

然后可以运行下面的代码，去查看模型的推理结果。

~~~python
img = XXX # 指定推理的图片路径
model = MMdetection(backbone='FasterRCNN') # 初始化模型
model.inference(is_trained=True,
                pretrain_model='./checkpoints/detection_model/latest.pth',
                img=img) # 进行推理
~~~

在这里，我们重点介绍`inference`这个函数的其他参数，可以看到和[直接推理](####1.直接推理)这一步的代码相比，`inference`中多了两个没见过的参数，分别是`is_trained`和`pretrain_model`，`is_trained`是一个布尔值，为`True`时代表从本地训练过的模型中进行推理，而`is_trained`设置为True之后则需要指定加载模型的路径，这也就是`pretrain_model`这个参数的作用，我们会从`pretrain_model`中加载模型进行推理，然后展示推理结果（如下图）。
![jupyter](img/det1.png)


#### 4.继续训练

在这一步中，我们会教您加载之前训练过的模型接着训练，如果您觉得之前训练的模型epoch数不够的话或者因为一些客观原因而不得不提前结束训练，相信下面的代码会帮到您。

~~~python
model = MMdetection(backbone='FasterRCNN') # 初始化实例模型
model.load_dataset(path=DATASET_PATH) # 配置数据集路径
model.num_classes = 1 # 进行车牌识别，此时只有一个类别。
model.save_fold = "checkpoints/det_model/" # 设置模型的保存路径
model.train(epochs=15,
            checkpoint='./checkpoints/det_model/latest.pth',
            validate=True) # 进行再训练
~~~

这里我们有一个参数在之前的[训练模型](####2.训练模型)过程中没有提及，那就是`train`函数中的`checkpoint`参数，这个放到这里就比较好理解，它的意思是指定需要进行再训练的模型路径，当然你也可以根据你需要训练的不同模型而调整参数。











