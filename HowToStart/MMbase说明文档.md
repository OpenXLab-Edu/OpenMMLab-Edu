## MMBase使用说明

### 0.引入包

```python
from MMBase import *
```

### 1.声明模型

```python
model = MMBase()
```

### 2.载入数据

此处采用随机生成的数据作为示例。

```python
# 随机生成数据
x = Variable(torch.randn(100, 1, 20,20))  # 训练数据，100张宽高为20*20的单通道图片
y = Variable(torch.randint(0, 2, (100,)))  # 训练标签， 100个[0，2）之间的标签
test_x = Variable(torch.randn(1,1,20,20)) # 测试数据
# 将数据载入
model.load_data(x, y)
```

### 3.搭建模型

逐层添加，搭建起模型结构，此处以典型的LeNet5网络结构为例。注释标明了数据经过各层的尺寸变化。

```python
model.add_layer('Conv', 1, 3, 3, 3) # [100, 3, 18, 18]
model.add_layer('MaxPool', 2) # [100, 3, 9, 9]
model.add_layer('Conv', 3, 10, 3, 3) # [100, 10, 7, 7]
model.add_layer('AvgPool', 2) # [100, 10, 3, 3]
model.add_layer('Reshape') # [100, 90]
model.add_layer('Linear', 90, 10) # [100, 10]
model.add_layer('ReLU') # [100, 90]
model.add_layer('Linear', 10, 2) # [100, 2]
model.add_layer('ReLU') # [100, 2]
model.add_layer('Softmax') # [100, 2]
```

添加层的方法为`add_layer(module, *size)`，参数module为层的类型，size为相关参数。

以下具体讲述各种层：

Conv：卷积层（二维），参数分别为输入维度，输出维度，卷积核宽，卷积核高。

MaxPool：最大池化层，参数为核的大小。

AvgPool：平均池化层，参数为核的大小。

Linear：线性层，参数为输入维度，输出维度。

ReLU：激活层，无参数。

Softmax：将结果归一化，无参数。

Reshape：将三维向量（通道 * 宽 * 高）形状调整为一维（即拉长），无参数。

### 4.模型训练

```python
model.train(lr=0.1, epochs=30)
```

参数lr为学习率， epochs为训练轮数。

### 5.模型推理

```python
model.inference(data=test_x)
```

参数data为待推理数据，传入测试数据即可进行推理。

### 6.模型的保存与加载

```python
# 保存
model.save("mmbase_net.pkl")
# 加载
model.load("mmbase_net.pkl")
```

参数为模型保存的路径，`.pkl`文件格式可以理解为将python中的数组、列表等持久化地存储在硬盘上的一种方式。

### 7.查看模型结构

```python
model.print_model()
```

无参数。



完整测试用例可见test.py文件。