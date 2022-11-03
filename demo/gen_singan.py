from base import *
from MMEdu import MMGeneration as gen


def only_infer_demo():
	# 指定一张图片
	img = 'demo/balloons.png'
	# 实例化模型，网络名称为'SinGAN'
	model = gen(backbone="SinGAN", )
	# 指定权重文件的路径
	checkpoint = '../checkpoints/gen_model/hand_gray/latest.pth'
	# 推理，并弹出窗口输出结果
	model.inference(is_trained=False, infer_data=img, pretrain_model = checkpoint, save_path = "../results/gen_result.jpg")


def normal_train_demo():
	# 实例化模型，网络名称为'SinGAN'
	model = gen(backbone='SinGAN')
	# 指定数据集的路径
	model.load_dataset(path='../dataset/gen/balloons.png')
	# 指定新训练模型的保存路径
	model.save_fold = "../checkpoints/gen"
	# 直接训练，“validate=True”表示每轮训练后，在验证集上测试一次准确率
	model.train(epoch=90, validate=True, inverse=True)
	# 训练结束后进行推理
	# model.inference(is_trained=True, 
	# 				pretrain_model = 'checkpoints/gen_model/ckpt/shoes2edges/latest.pth', 
	# 				infer_data= 'demo/184_AB.jpg',
	# 				save_path = "results/gen_result.jpg")

def continue_train_demo():
	# 实例化模型，网络名称为'SinGAN'
	model = gen(backbone='SinGAN')
	# 指定数据集的路径
	model.load_dataset(path='balloons.png')
	# 指定新训练模型的保存路径
	model.save_fold = "../checkpoints/gen"
	# 指定预训练模型的权重文件
	checkpoint='../checkpoints/gen/singan_balloons_20210406_191047-8fcd94cf.pth'
	# 在预训练权重文件的基础上继续训练，“validate=True”表示每轮训练后，在验证集上测试一次准确率
	model.train(epoch=15, pretrain_model = checkpoint, validate=True, inverse=True)


if __name__ == "__main__":
	# 请按照次序逐一调用函数，训练模型要耗费较长时间。MMEdu的文档，可以获得更多帮助。
	only_infer_demo()
	# normal_train_demo()
	# continue_train_demo()
