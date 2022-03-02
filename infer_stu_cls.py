from turtle import back
from utils.my_utils_new import MMClassification


def test1():
	img = 'utils/demo/bird.JPEG'
	model = MMClassification()
	result = model.inference(image=img)
	print(result)


def test2():
	img = 'utils/demo/bird.JPEG'
	model = MMClassification(backbone='MobileNet')
	result = model.inference(image=img)
	print(result)


def test3():
	# 刚刚查到mmcv不支持cfg回传保存成config,需要我们自己魔改,因此3被搁置
	# 思路是3才会调用train,调用完了之后输出一个config.py文件,让学生自己保存到外面然后进行一次2的步骤.
	model = MMClassification(backbone='MobileNet')
	model.num_classes = 2
	model.load_dataset(path='data/ImageNet')
	model.train(epochs=1, device='cuda:0', validate=False)
	model.inference(is_trained=True, pretrain_model = './checkpoints/latest.pth',image='./data/ImageNet/test_set/test_set/cats/cat.4003.jpg')


if __name__ == "__main__":
	# test1()
	# test2()
	test3()
