import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        # print(x.shape,x.view(x.shape[0], -1).shape)
        return x.view(x.shape[0], -1)

class MMBase:
    def __init__(self):
        self.model = nn.Sequential()
        self.batchsize = None
        self.layers = []
        self.layers_num = 0
        self.x = None
        self.y = None

    def add_layer(self, module, *size):
        self.layers_num += 1
        self.layers.append(module)
        if module == 'Linear':
            self.model.add_module('Reshape', Reshape(self.batchsize))
            self.model.add_module('Linear' + str(self.layers_num), nn.Linear(size[0], size[1]))
            print("增加全连接层，输入维度:{},输出维度：{}。".format(size[0], size[1]))
        elif module == 'Reshape':
            self.model.add_module('Reshape', Reshape(self.batchsize))
        elif module == 'ReLU':
            self.model.add_module('ReLU' + str(self.layers_num), nn.ReLU())
            print("增加ReLU层。")
        elif module == 'Conv':
            self.model.add_module('Conv' + str(self.layers_num), nn.Conv2d(size[0], size[1], (size[2], size[3])))
            print("增加卷积层，输入维度:{},输出维度：{},kernel_size: {} * {}".format(size[0], size[1], size[2], size[3]))
        elif module == 'MaxPool':
            self.model.add_module('MaxPooling' + str(self.layers_num), nn.MaxPool2d(size[0]))
            print("增加最大池化层,kernel_size: {} * {}".format(size[0], size[0]))
        elif module == 'AvgPool':
            self.model.add_module('MaxPooling' + str(self.layers_num), nn.AvgPool2d(size[0]))
            print("增加平均池化层,kernel_size: {} * {}".format(size[0], size[0]))
        # self.model.add_module('Layer' + str(self.layers_num), self.modules[module](size))
        elif module == 'Softmax':
            self.model.add_module('Softmax'+str(self.layers_num), nn.Softmax())
            print('增加Softmax层。')

    def load_data(self, x, y):
        self.x = x
        self.y = y
        self.batchsize = self.x.shape[0]

    def train(self, lr=0.1, epochs=30):
        loss_fun = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)  # 使用SGD优化器
        for epoch in range(epochs):  # 10个epoch
            y_pred = self.model(self.x)
            loss = loss_fun(y_pred, self.y)
            print("{epoch:%d  Loss:%.4f}" % (epoch, loss))
            optimizer.zero_grad()  # 将梯度初始化为零，即清空过往梯度
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数

    def inference(self, data):
        res = self.model(data)
        print("推理结果为：",res)
        return res

    def print_model(self):
        # print('模型共{}层'.format(self.layers_num))
        print(self.model)

    def save(self, model_path='mmbase_net.pkl'):
        print("保存模型中...")
        torch.save(self.model, model_path)
        print("保存模型{}成功！".format(model_path))
    
    def load(self,model_path):
        print("载入模型中...")
        self.model = torch.load(model_path)
        print("载入模型{}成功！".format(model_path))

def test1():
    x = Variable(torch.randn(100, 1, 20,20))  # 随机数据
    y = Variable(torch.randint(0, 2, (100,)))  # 随机标签
    test_x = Variable(torch.randn(1,1,20,20)) # 测试数据
    model = MMBase()
    model.load_data(x, y)
    model.add_layer('Conv', 1, 3, 3, 3) # [100, 3, 18, 18]
    model.add_layer('MaxPool', 2) # [100, 3, 9, 9]
    model.add_layer('Conv', 3, 10, 3, 3) # [100, 10, 7, 7]
    model.add_layer('AvgPool', 2) # [100, 10, 3, 3]
    model.add_layer('Reshape') # [100, 90]
    model.add_layer('Linear', 90, 10) # [100, 10]
    model.add_layer('ReLU')
    model.add_layer('Linear', 10, 2)
    model.add_layer('ReLU')
    model.add_layer('Softmax')
    # model.print_model()
    model.train(lr=0.1, epochs=30)
    model.save("mmbase_net.pkl")
    model.load("mmbase_net.pkl")
    model.inference(test_x)

if __name__ == "__main__":
    test1()
