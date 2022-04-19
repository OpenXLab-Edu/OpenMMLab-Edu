from base import *
from MMEdu import MMClassification


def only_infer_demo():
	img = 'dog.jpg'
	model = MMClassification(backbone='MobileNet')
	model.checkpoint = '../checkpoints/cls_model/cats_dogs/latest.pth'
	result = model.inference(image=img, show=True)
	print(result)


def continue_train_demo():
	model = MMClassification(backbone='MobileNet')
	model.num_classes = 2
	model.save_fold = '../checkpoints/cls_model/cats_dogs'
	model.load_dataset(path='../dataset/cls/cats_dogs_dataset')
	model.train(epochs=1, validate=True, checkpoint='../checkpoints/cls_model/cats_dogs/latest.pth')


def normal_train_demo():
	model = MMClassification(backbone='MobileNet')
	model.num_classes = 2
	model.save_fold = '../checkpoints/cls_model/cats_dogs'
	model.load_dataset(path='../dataset/cls/cats_dogs_dataset')
	model.train(epochs=100, validate=True)


if __name__ == "__main__":
	only_infer_demo()
	# continue_train_demo()
	# normal_train_demo()