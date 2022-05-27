from base import *
from MMEdu import MMGeneration


def only_infer_demo():
	img = 'demo/balloons.png'
	model = MMGeneration(backbone="SinGAN")
	model.inference(is_trained=False, infer_data=img, save_path = "../results/gen_result.jpg")


def normal_train_demo():
	model = MMGeneration(backbone='SinGAN')
	model.load_dataset(path='../dataset/gen/balloons.png')
	model.save_fold = "../checkpoints/gen"
	model.train(epoch=90, validate=True, inverse=True)
	# model.inference(is_trained=True, 
	# 				pretrain_model = 'checkpoints/gen_model/ckpt/shoes2edges/latest.pth', 
	# 				infer_data= 'demo/184_AB.jpg',
	# 				save_path = "results/gen_result.jpg")

def continue_train_demo():
	model = MMGeneration(backbone='SinGAN')
	model.load_dataset(path='balloons.png')
	model.save_fold = "../checkpoints/gen"
	model.train(epoch=15, checkpoint='../checkpoints/gen/singan_balloons_20210406_191047-8fcd94cf.pth', validate=True, inverse=True)


if __name__ == "__main__":
	only_infer_demo()
	# normal_train_demo()
	# continue_train_demo()
