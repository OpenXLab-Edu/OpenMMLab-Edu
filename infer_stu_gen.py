from utils.my_utils_gen import MMGeneration


def only_infer_demo():
	img = 'data/edges2shoes/val/2_AB.jpg'
	model = MMGeneration(backbone="Pix2Pix", dataset_path="data/edges2shoes")
	model.inference(is_trained=False, infer_data=img, save_path = "result/demo.png")

def simple_train_demo():
	model = MMGeneration()
	model.load_dataset(path='data/edges2shoes')
	model.save_fold = "checkpoints/gen_edges2shoes"
	model.train(total_iters=5000)


def normal_train_demo():
	model = MMGeneration(backbone='Pix2Pix')
	model.load_dataset(path='data/edges2shoes')
	model.save_fold = "checkpoints/gen_edges2shoes"
	# model.train(iters=5000, validate=True)
	model.inference(is_trained=True, 
					pretrain_model = 'checkpoints/gen_edges2shoes/ckpt/gen_edges2shoes/latest.pth', 
					infer_data='utils/demo/2_AB.jpg', 
					save_path = "result/2_AB.jpg")


# def continue_train_demo():
# 	model = MMDetection(backbone='FasterRCNN')
# 	model.num_classes = 1
# 	model.load_dataset(path='data/coco/')
# 	model.save_fold = "checkpoints/det_plate"
# 	model.train(epochs=15, checkpoint='./checkpoints/det_plate/latest.pth', validate=True, Frozen_stages=1)


if __name__ == "__main__":
	# only_infer_demo()
	# simple_train_demo()
	normal_train_demo()
	# continue_train_demo()
