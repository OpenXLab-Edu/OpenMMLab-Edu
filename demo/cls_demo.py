from base import *
from MMEdu import MMClassification as cls


def only_infer_demo():
	img = 'dog.jpg'
	model = cls(backbone='ResNet18')
	model.num_classes = 2
	model.checkpoint = '../checkpoints/cls_model/cats_dogs/latest.pth'
	result = model.inference(image=img, show=True)
	print(result)


def continue_train_demo():
	model = cls(backbone='ResNet18')
	model.num_classes = 2
	model.save_fold = '../checkpoints/cls_model/cats_dogs'
	model.checkpoint = '../checkpoints/cls_model/cats_dogs/latest.pth'
	model.load_dataset(path='../dataset/cls/cats_dogs')
	model.train(epochs=1, validate=True, checkpoint=model.checkpoint)
	result = model.inference(image='dog.jpg', class_path='../dataset/cls/cats_dogs/classes.txt', checkpoint=model.checkpoint)
	print(result)


def normal_train_demo():
	model = cls(backbone='ResNet18')
	model.num_classes = 2
	model.save_fold = '../checkpoints/cls_model/cats_dogs'
	model.load_dataset(path='../dataset/cls/cats_dogs')
	model.train(epochs=250, validate=True)
	result = model.inference(image='dogs.jpg')
	print(result)


if __name__ == "__main__":
	# only_infer_demo()
	continue_train_demo()
	# normal_train_demo()