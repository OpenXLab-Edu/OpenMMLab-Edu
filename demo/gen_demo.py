from base import *
from MMEdu import MMGeneration as gen


def only_infer_demo():
	# 指定一张图片
	img = '184_AB.jpg'
	# 实例化模型，网络名称为'Pix2Pix'
	model = gen(backbone="Pix2Pix", dataset_path="../dataset/gen_model/edges2shoes")
	# 指定权重文件的路径
	checkpoint = '../checkpoints/gen_model/hand_gray/latest.pth'
	# 推理，并弹出窗口输出结果
	result = model.inference(infer_data=img, pretrain_model = checkpoint, save_path = "../results/gen_result.jpg")
	print(result)


def normal_train_demo():
	# 实例化模型，网络名称为'Pix2Pix'
	model = gen(backbone='Pix2Pix')
	# 指定数据集的路径
	model.load_dataset(path='../dataset/gen/edges2shoes')
	# 指定新训练模型的保存路径
	model.save_fold = "../checkpoints/gen_model"
	# 直接训练，“validate=True”表示每轮训练后，在验证集上测试一次准确率
	model.train(epochs=50, validate=True, inverse=False)
	# 训练结束后进行推理
	# model.inference(pretrain_model = '../checkpoints/gen_model/ckpt/gen_model/latest.pth', 
	# 				infer_data= '184_AB.jpg',
	# 				save_path = "../results/gen_result.jpg")


def continue_train_demo():
	# 实例化模型，网络名称为'Pix2Pix'
	model = gen(backbone='Pix2Pix')
	# 指定数据集的路径
	model.load_dataset(path='../dataset/edges2shoes')
	# 指定新训练模型的保存路径
	model.save_fold = "../checkpoints/gen_model"
	# 指定预训练模型的权重文件
	checkpoint = '../checkpoints/gen_model/ckpt/gen_model/latest.pth'
	# 在预训练权重文件的基础上继续训练，“validate=True”表示每轮训练后，在验证集上测试一次准确率
	model.train(epochs=15, checkpoint=checkpoint, validate=True, inverse=True)


if __name__ == "__main__":
	# 请按照次序逐一调用函数，训练模型要耗费较长时间。MMEdu的文档，可以获得更多帮助。
	only_infer_demo()
	# normal_train_demo()
	# continue_train_demo()
