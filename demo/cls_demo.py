from MMEdu import MMClassification


def only_infer_demo():
	img = 'bird.JPEG'
	model = MMClassification(backbone='MobileNet')
	model.checkpoint = 'checkpoints/cls_model/MobileNet/MobileNet.pth'
	result = model.inference(image=img)
	print(result)


def simple_train_demo():
	model = MMClassification()
	model.load_dataset(path='dataset/fruit_dataset')
	model.train(epochs=50, validate=False)
	# 以下代码可测试训练出的模型的效果
	# model.inference(is_trained=True, image='fruit_dataset/test/apple_test_1.jpg')


def normal_train_demo():
	model = MMClassification(backbone='MobileNet')
	model.num_classes = 3
	model.save_fold = 'checkpoints/cls_model/fruit'
	model.load_dataset(path='dataset/fruit_dataset')
	model.train(epochs=50, validate=False)
	# 以下代码可测试训练出的模型的效果
	# model.inference(is_trained=True, pretrain_model='new_checkpoints/latest.pth', image='fruit_dataset/test/banana_test_1.jpg')


def continue_train_demo():
	model = MMClassification(backbone='MobileNet')
	model.num_classes = 3
	# model.save_fold = 'checkpoints/cls_fruit'
	model.load_dataset(path='fruit_dataset')
	model.train(epochs=5, validate=False, checkpoint='checkpoints/cls_model/fruit/latest.pth')


if __name__ == "__main__":
	# only_infer_demo()
	# simple_train_demo()
	# normal_train_demo()
	continue_train_demo()
