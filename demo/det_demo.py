from base import *
from MMEdu import MMDetection


def only_infer_demo():
	img = 'car_plate.png'
	model = MMDetection(backbone="FasterRCNN", dataset_path='../dataset/det/coco')
	model.inference(infer_data=img, show=True, rpn_threshold=0.7, rcnn_threshold=0.7)


def continue_train_demo():
	model = MMDetection(backbone='FasterRCNN')
	model.num_classes = 1
	model.load_dataset(path='../dataset/det/coco')
	model.save_fold = "../checkpoints/det_model/plate"
	model.train(epochs=3, checkpoint='../checkpoints/det_model/plate/latest.pth', validate=True, Frozen_stages=1)


def normal_train_demo():
	model = MMDetection(backbone='FasterRCNN')
	model.num_classes = 1
	model.load_dataset(path='../dataset/det/coco')
	model.save_fold = "../checkpoints/det_model/plate"
	model.train(epochs=100, validate=True, Frozen_stages=1)


if __name__ == "__main__":
	only_infer_demo()
	# continue_train_demo()
	# normal_train_demo()