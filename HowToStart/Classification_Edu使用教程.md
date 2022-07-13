# Classification_Edu使用教程

### Classification_Edu简介

Classification_Edu是MMEdu团队在基于OpenMMlab项目的MMClassification模块研发的教育版工具，在实现MMClassification原始功能的基础上，框架更加简单，结构更加清晰，使用难度也随之降低，适用于课程教学、敏捷开发、迁移开发和新手入门。

主要功能：对图片进行分类。

### Classification_Edu使用说明

测试版中，我们预置了猫狗二分类的模型，并在demo文件夹中放了猫狗各一张图片。

#### 1.直接推理

如果想快速上手体验Classification_Edu的话，我们建议您使用我们自带的图片去进行单独的推理。

执行代码如下:

~~~python
import base # 测试版需要（发布版不需要）
from MMEdu import MMClassification  # 导入mmcls模块

img='cat.jpg' # 指定进行推理的图片路径，我们使用demo文件夹中提供的图片
model=MMClassification(backbone='MobileNet') # 实例化mmcls模型
model.checkpoint='../checkpoints/cls_model/cats_dogs/latest.pth' # 指定使用的模型权重文件
model.inference(img=img, device='cpu', show=True) # 在CPU上进行推理
# 同时您可以修改show的值来决定是否需要显示结果图片，此处默认显式结果图片
~~~

运行结果如图：

<img src="C:\Users\Jx_G\Desktop\mmedu_base\mmedu_base\tutorials\imgs\image-20220408185534541.png" alt="image-20220408185534541" style="zoom:45%;" />

接下来对为您讲述代码规则

- **图片准备**：

~~~python
img = 'cat.jpg' # 指定推理图片的路径，直接在代码所在的demo文件夹中选择图片
~~~

如果使用自己的图片的话，只需要修改img的路径即可（绝对路径和相对路径均可）

- **实例化模型**：

~~~python
model = MMClassification(backbone='MobileNet') # 实例化mmcls模型
~~~

这里对于`MMClassification`模型提供的参数进行解释，`MMClassification`支持传入三个参数：分别是`backbone`和`dataset_path`。

`backbone`：指定使用的MMClassification模型，默认参数是`'MobilNet'`，当然读者可以自行修改该参数以使用不同模型。

`dataset_path`：指定训练集的路径，默认参数是`None`，后面用到时会详细介绍。

- **模型推理**：

~~~python
model.inference(img=img,device='cpu') # 在cpu上进行推理
~~~

将所需要推理图片的路径传入`inference`函数中即可进行推理，我们这里传入了两个参数，`img`代表的就是推理图片的路径，`device`是推理所用的设备，默认是参数是`'cuda:0'`，使用GPU，我们这里将其赋值为`'cpu'`，给设备不支持GPU的同学做一个参考。

在Classification_Edu中对于`inference`函数还有其他的传入参数，在这里进行说明：

`show`：布尔值，默认为`True`，表示推理后是否显示推理结果。

`save`：布尔值，默认为`True`，表示对于推理的结果图片是否保存到本地。

`word_dir`：推理结果图片的保存路径，默认参数为`'./'`即当前目录下，用户可在函数中设置该参数来调整图片的保存路径。

`name`：保存的图片名，数据结构为字符串，默认参数为`'cls_result'`，用户也可以定义为自己想要的名字。

#### 2.训练模型

使用下面的代码即可简单体验Classification_Edu的训练过程，我们会为您进行详细的介绍。

在运行代码之前，您需要先拥有一个数据集，这里我们为您提供经典的猫狗分类数据集。

数据集文件结构如下:

![image-20220408163050392](imgs\image-20220408163050392.png)

**cats_dogs_dataset**文件夹中包含三个图片文件夹，`test_set`,`training_set`,`val_set`分别存储测试集，训练集和验证集的图片；以及三个`txt`文件，`classes.txt`记录该数据集的类别，`test.txt`和`val.txt`分别记录测试集和验证集的图片名。

- 代码展示

~~~python
model = MMClassification() # 实例化模型，不指定参数即使用默认参数。
model.load_dataset(path='../dataset/cls/cats_dogs_dataset') # 从指定数据集路径中加载数据
model.train(epochs=5, validate=True) # 设定训练的epoch次数以及是否进行评估
~~~

**详细说明**

实例化模型的代码在前面说过就不再赘述。

- 加载数据集：

~~~python
model.load_dataset(path='../dataset/cls/cats_dogs_dataset') # 从指定数据集路径中加载数据
~~~

这个函数只需要传入一个`path`参数即训练数据集的路径，函数的作用是修改模型中关于数据集路径的配置文件，从而确保我们在训练时不会找错文件。

这里补充一点：

~~~python
model = MMClassification() # 实例化模型，不指定参数即使用默认参数。
model.load_dataset(path='../dataset/cls/cats_dogs_dataset') # 从指定数据集路径中加载数据
~~~

上面两行代码和`model = MMClassification(dataset_path='../dataset/cls/cats_dogs_dataset')`这一行代码所实现的结果是一样的。

而我们设置不同的函数和接口实现这个功能是为了确保适用于不同的场景，比如说`model = MMClassification(dataset_path='../dataset/cls/cats_dogs_dataset')`这行代码只能在模型的初始化阶段指定数据集路径，而`model.load_dataset(path='../dataset/cls/cats_dogs_dataset')`可以在模型初始化之后的任何阶段进行修改，比如我可以先用猫狗数据集进行一个阶段的训练，然后更换为其他数据集进行下一轮训练。

- 模型训练

~~~python
model.train(epochs=5, validate=True) # 设定训练的epoch次数以及是否进行评估
~~~

其实这一部分的参数比较多，我先解释一些好理解的以及常用的参数。

`epochs`：默认参数为`100`，用于指定训练的轮次，而在上述代码中我们设置为`5`。

`validate`：布尔值，只能为`True`或者`False`，默认参数为`True`，在训练结束后，设定是否需要在校验集上进行评估，`True`则是需要进行评估。

`lr`：学习率，默认参数为`5e-4`即`0.0005`，指定模型进行梯度下降时的步长。简单解释就是，学习率过小，训练过程会很缓慢，学习率过大时，模型精度会降低。

`save_fold`：模型的保存路径，默认参数为`./checkpoints/cls_model/`，如果不想模型保存在该目录下，可自己指定路径。

`optimizer`：进行迭代时的优化器，默认参数为`Adam`，`Adam`会在训练的过程中动态调整学习率，避免学习率过小或过大。

`metric`：验证指标，默认参数为`'accuracy'`，在进行模型评估时会计算分类准确率，数值越高说明模型性能越好，我们在运行完程序之后也会看到这个结果。

`random_seed`：随机种子策略，默认为`0`即不使用，使用随机种子策略会减小模型算法结果的随机性。



执行上述代码之后的运行结果如下图

<img src="C:\Users\Jx_G\Desktop\mmedu_base\mmedu_base\tutorials\imgs\image-20220408185806266.png" alt="image-20220408185806266" style="zoom:60%;" />

而在`checkpoints\cls_model`文件夹中我们会发现多了两种文件，一个是`None.log.json`文件，它记录了我们模型在训练过程中的一些参数，比如说学习率`lr`，所用时间`time`，以及损失`loss`等；另一个文件是.pth文件，这个是我们在训练过程中所保存的模型。



#### 3.使用自己训练的模型进行推理

这一步像是把推理和训练结合到了一起，首先**确保你进行了推理和训练的准备工作**，并且使用训练的代码在本地保存了一个模型。

然后可以运行下面的代码，去查看模型的推理结果。

~~~python
img = 'dog.jpg' # 指定推理的图片路径
model = MMClassification(backbone='MobileNet') # 初始化模型
model.inference(is_trained=True,
                pretrain_model='../checkpoints/cls_model/cats_dogs/latest.pth',
                img=img) # 进行推理
~~~

在这里，我们重点介绍`inference`这个函数的其他参数，可以看到和[直接推理](####1.直接推理)这一步的代码相比，`inference`中多了两个没见过的参数，分别是`is_trained`和`pretrain_model`，`is_trained`是一个布尔值，为`True`时代表从本地训练过的模型中进行推理，而`is_trained`设置为True之后则需要指定加载模型的路径，这也就是`pretrain_model`这个参数的作用，我们会从`pretrain_model`中加载模型进行推理，然后展示推理结果（如下图）。

<img src="C:\Users\Jx_G\Desktop\mmedu_base\mmedu_base\tutorials\imgs\image-20220408190008470.png" alt="image-20220408190008470" style="zoom:50%;" />



#### 4.继续训练

在这一步中，我们会教您加载之前训练过的模型接着训练，如果您觉得之前训练的模型epoch数不够的话或者因为一些客观原因而不得不提前结束训练，相信下面的代码会帮到您。

~~~python
model = MMClassification(backbone='MobileNet') # 初始化实例模型
model.load_dataset(path='../dataset/cls/cats_dogs_dataset') # 配置数据集路径
model.save_fold = '../checkpoints/cls_model/cats_dogs' # 设置模型的保存路径
model.train(epochs=15,
            checkpoint='../checkpoints/cls_model/cats_dogs/latest.pth'
            validate=True) # 进行再训练
~~~

这里我们有一个参数在之前的[训练模型](####2.训练模型)过程中没有提及，那就是`train`函数中的`checkpoint`参数，这个放到这里就比较好理解，它的意思是指定需要进行再训练的模型路径，当然你也可以根据你需要训练的不同模型而调整参数。

