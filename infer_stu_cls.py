from utils.my_utils_cls import MMClassification


def only_infer_demo():
	img = 'utils/demo/bird.JPEG'
	model = MMClassification(backbone='MobileNet')
	result = model.inference(image=img)
	print(result)


def simple_train_demo():
	model = MMClassification()
	model.load_dataset(path='fruit_dataset')
	model.train(epochs=50, validate=False)
	# 以下代码可测试训练出的模型的效果
	# model.inference(is_trained=True, image='fruit_dataset/test/apple_test_1.jpg')


def normal_train_demo():
	model = MMClassification(backbone='MobileNet')
	model.num_classes = 3
	model.save = 'new_checkpoints/'
	model.load_dataset(path='fruit_dataset')
	model.train(epochs=50, validate=False)
	# 以下代码可测试训练出的模型的效果
	# model.inference(is_trained=True, pretrain_model='new_checkpoints/latest.pth', image='fruit_dataset/test/banana_test_1.jpg')


def continue_train_demo():
	model = MMClassification(backbone='MobileNet')
	model.num_classes = 3
	# model.save = 'new_checkpoints/'
	model.load_dataset(path='fruit_dataset')
	model.train(epochs=5, validate=False, checkpoint='checkpoints/latest.pth')


if __name__ == "__main__":
	# only_infer_demo()
	# simple_train_demo()
	# normal_train_demo()
	continue_train_demo()