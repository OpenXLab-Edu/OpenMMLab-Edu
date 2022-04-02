from MMEdu import MMSegmentation

def only_infer_demo():
	img = r'Image_11L.png'
	model = MMSegmentation(backbone='UNet')
	result = model.inference(image=img)
	print(result)


def simple_train_demo():
	model = MMSegmentation()
	model.load_dataset(path='data/')
	model.train(epochs=60000, validate=False)
	# 以下代码可测试训练出的模型的效果
	#model.inference(is_trained=True, image=r'Image_11L.png')


def normal_train_demo():
	model = MMSegmentation(backbone='UNet')
	model.num_classes = 19
	model.save = 'new_checkpoints/'
	model.load_dataset(path='data/')
	model.train(epochs=40000, validate=False)
	# 以下代码可测试训练出的模型的效果
	# model.inference(is_trained=True, pretrain_model='new_checkpoints/latest.pth', image='fruit_dataset/test/banana_test_1.jpg')


def continue_train_demo():
	model = MMSegmentation(backbone='UNet')
	model.num_classes = 19
	model.save = 'new_checkpoints/'
	model.load_dataset(path='')
	model.train(epochs=10, validate=False, checkpoint='checkpoints/latest.pth')


if __name__ == "__main__":
	# only_infer_demo()
	simple_train_demo()
	# normal_train_demo()
	# continue_train_demo()
