from base import *
from MMEdu import MMGeneration


def only_infer_demo():
	img = '184_AB.jpg'
	model = MMGeneration(backbone="Pix2Pix", dataset_path="../dataset/gen_model/edges2shoes")
	model.inference(infer_data=img, save_path = "../results/gen_result.jpg")


def normal_train_demo():
	model = MMGeneration(backbone='Pix2Pix')
	model.load_dataset(path='../dataset/gen/edges2shoes')
	model.save_fold = "../checkpoints/gen_model"
	model.train(epochs=50, validate=True, inverse=False)
	model.inference(pretrain_model = '../checkpoints/gen_model/ckpt/gen_model/latest.pth', 
					infer_data= '184_AB.jpg',
					save_path = "../results/gen_result.jpg")

def continue_train_demo():
	model = MMGeneration(backbone='Pix2Pix')
	model.load_dataset(path='../dataset/edges2shoes')
	model.save_fold = "../checkpoints/gen_model"
	model.train(epochs=15, checkpoint='../checkpoints/gen_model/ckpt/gen_model/latest.pth', validate=True, inverse=True)


if __name__ == "__main__":
	only_infer_demo()
	# normal_train_demo()
	# continue_train_demo()
