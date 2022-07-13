from base import *
from BaseEdu import *
import numpy as np

def cal_accuracy(y, pred_y):
    res = pred_y.argmax(dim=1)
    tp = np.array(y)==np.array(res)
    acc = np.sum(tp)/ y.shape[0]
    return acc

# 训练数据
train_path = '../dataset/iris/iris_training.csv'
x = np.loadtxt(train_path, dtype=float, delimiter=',',skiprows=1,usecols=range(0,4)) # [120, 4]
y = np.loadtxt(train_path, dtype=int, delimiter=',',skiprows=1,usecols=4)
# 测试数据
test_path = '../dataset/iris/iris_test.csv'
test_x = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=range(0,4))
test_y = np.loadtxt(test_path, dtype=int, delimiter=',',skiprows=1,usecols=4)
# 声明模型
model = BaseNet()
model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]
model.load_data(x, y)
# model.print_model()
model.train(lr=0.01, epochs=5000)
# model.save("mmbase_net.pkl")
# model.load("mmbase_net.pkl")
res = model.inference(test_x)
# 计算分类正确率
print("分类正确率为：",cal_accuracy(test_y, res))
