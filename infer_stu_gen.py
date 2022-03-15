from re import T
from torch import inverse
from utils.my_utils_gen import MMGeneration


def only_infer_demo():
	img = 'utils/demo/balloons.png'
	model = MMGeneration(backbone="SinGAN", tpye ="", dataset_path="data/edges2shoes")
	model.inference(is_trained=False, infer_data=img, save_path = "result/unconditional.jpg")


def simple_train_demo():
	model = MMGeneration()
	model.load_dataset(path='data/edges2shoes')
	# 这里尝试一下反向任务
	model.save_fold = "checkpoints/shoes2edges"
	model.train(epoch=50)


def normal_train_demo():
	model = MMGeneration(backbone='Pix2Pix')
	model.load_dataset(path='data/edges2shoes')
	model.save_fold = "checkpoints/shoes2edges"
	model.train(epoch=50, validate=True, inverse=True)
	model.inference(is_trained=True, 
					pretrain_model = 'checkpoints/shoes2edges/ckpt/shoes2edges/latest.pth', 
					infer_data='utils/demo/184_AB.jpg', 
					save_path = "result/184_AB.jpg")

def continue_train_demo():
	model = MMGeneration(backbone='Pix2Pix')
	model.load_dataset(path='data/edges2shoes')
	model.save_fold = "checkpoints/shoes2edges"
	model.train(epoch=15, checkpoint='checkpoints/shoes2edges/ckpt/shoes2edges/latest.pth', validate=True, inverse=True)


if __name__ == "__main__":
	# only_infer_demo()
	simple_train_demo()
	# normal_train_demo()
	# continue_train_demo()
