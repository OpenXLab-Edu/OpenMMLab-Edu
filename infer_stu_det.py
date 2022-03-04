from re import T
from turtle import back
from utils.my_utils_det import MMDetection
from utils.my_utils_cls import MMClassification


def test1():
	img = 'utils/demo/bird.JPEG'
	model = MMDetection()
	result = model.inference(infer_data=img)
	print(result)


def test2():
	img = 'utils/demo/bird.JPEG'
	model = MMDetection(backbone='Faster-RCNN')
	result = model.inference(infer_data=img)
	print(result)


def test3():
	model = MMDetection(backbone='FasterRCNN',dataset_path='data/coco/')
	model.load_dataset(path='data/coco/')
	# model.train(epochs=15, validate=True, Frozen_stages=1)
	model.inference(is_trained=True, pretrain_model = './checkpoints/latest.pth',infer_data='./data/coco/images/test/0001.jpg', iou_threshold=0.3)


if __name__ == "__main__":
	# test1()
	# test2()
	test3()
