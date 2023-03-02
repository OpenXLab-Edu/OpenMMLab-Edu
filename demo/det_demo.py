# from cgitb import reset
# from statistics import mode
# from base import *
from MMEdu import MMDetection as det

# 使用预训练的模型对图片进行推理，需要知道模型的名称和权重文件位置
def only_infer_demo():
	# 指定一张图片
	img = '/home/user/桌面/pip测试9/dataset/det/coco/images/test/0001.jpg'
	# 实例化模型，网络名称为“FasterRCNN”
	model = det(backbone='FasterRCNN')
	# 指定权重文件的路径，代表训练集中所包含的所有类别
	checkpoint = '../../checkpoints/det_model/plate_111/latest.pth'
	# 指定训练集的路径，代表训练集中所包含的所有类别
	# 推理，“show=True”表示弹出识别结果窗口
	result = model.inference(image=img, show=True,checkpoint = checkpoint)
	# 输出结果，将inference函数输出的结果修饰后输出具体信息
	print(result)
	model.print_result(result)

# 在预训练基础上继续训练
def continue_train_demo():
	# 实例化模型，网络名称为“FasterRCNN”
	model = det(backbone='FasterRCNN')
	# 指定识别目标的类别数量
	model.num_classes = 1
	# 指定数据集的路径
	model.load_dataset(path='../dataset/det/coco')
	# 指定新训练模型的保存路径
	model.save_fold = '../../checkpoints/det_model/plate_continue'
	# 指定预训练模型的权重文件
	checkpoint='../checkpoints/det_model/plate/latest.pth'
	# 在预训练权重文件的基础上继续训练，“validate=True”表示每轮训练后，在验证集上测试一次准确率
	model.train(epochs=3, validate=True, checkpoint=checkpoint)

# 从零开始训练模型
def normal_train_demo():
	# 实例化模型，网络名称为“FasterRCNN”
	model = det(backbone='FasterRCNN')
	# 识别目标的类别数量
	model.num_classes = 1
	# 指定数据集的路径
	model.load_dataset(path='../../dataset/det/coco')
	# 指定保存模型配置文件和权重文件的路径
	model.save_fold = '../../checkpoints/det_model/fashion111'
	# 开始训练，轮次为3，“validate=True”表示每轮训练后，在验证集上测试一次准确率
	model.train(epochs=3, validate=True, device='cpu')
	# 训练结束后，可以用“model.inference”进行推理，看看效果。

def fast_infer_demo():
	# 指定一张图片
	img = 'car_plate.png'
	# 实例化模型，网络名称为“FasterRCNN”
	model = det(backbone='FasterRCNN')
	# 指定权重文件的路径，代表训练集中所包含的所有类别
	checkpoint = '../checkpoints/det_model/plate/latest.pth'
	# 推理，“show=True”表示弹出识别结果窗口
	model.load_checkpoint(checkpoint=checkpoint, device='cpu')
	import time 
	begin = time.time()
	for i in range(10):
		result = model.fast_inference(image=img,save_fold='det_result')
		model.print_result(result)
	print(time.time()-begin)
	# result = model.inference(image=img, show=True, class_path=class_path,checkpoint = checkpoint)

def convert_demo():
	model = det('SSD_Lite')
	model.num_classes = 80
	checkpoint = '../checkpoints/det_model/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth'
	out_file="convert_model.onnx"
	model.convert(checkpoint=checkpoint, backend="ONNX", out_file= out_file)

if __name__ == "__main__":
	# 请按照次序逐一调用函数，训练模型要耗费较长时间。MMEdu的文档，可以获得更多帮助。
	# normal_train_demo()
	only_infer_demo()
	# continue_train_demo()
	# fast_infer_demo()
	# convert_demo()
