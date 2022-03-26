from MMEdu.Detection_Edu import MMDetection


def only_infer_demo():
	img = 'data/coco/images/train/00020.jpg'
	model = MMDetection(backbone="FasterRCNN", dataset_path="data/coco")
	model.inference(infer_data=img, rpn_threshold=0.5, rcnn_threshold=0.3)

def simple_train_demo():
	model = MMDetection()
	model.num_classes = 1
	model.load_dataset(path='data/coco/')
	model.train(epochs=15)


def normal_train_demo():
	model = MMDetection(backbone='FasterRCNN')
	model.num_classes = 1
	model.load_dataset(path='data/coco/')
	model.save_fold = "checkpoints/det_plate"
	model.train(epochs=15, validate=True, Frozen_stages=1)
	model.inference(is_trained=True, pretrain_model = './checkpoints/det_plate/latest.pth', infer_data='./data/coco/images/test/0000.jpg', rpn_threshold=0.5, rcnn_threshold=0.3)


def continue_train_demo():
	model = MMDetection(backbone='FasterRCNN')
	model.num_classes = 1
	model.load_dataset(path='data/coco/')
	model.save_fold = "checkpoints/det_plate"
	model.train(epochs=15, checkpoint='./checkpoints/det_plate/latest.pth', validate=True, Frozen_stages=1)


if __name__ == "__main__":
	# only_infer_demo()
	# simple_train_demo()
	# normal_train_demo()
	continue_train_demo()
