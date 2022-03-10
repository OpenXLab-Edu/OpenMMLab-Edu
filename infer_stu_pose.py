from utils.my_utils_pose import MMPose

DATASET_PATH = '/home/PJLAB/congpeishan/Desktop/mmlab-proj/mmpose/demo/data/coco_tiny'
def only_infer_demo():
	img = 'utils/demo/000000000785.jpg'
	model = MMPose(backbone='HrNet')
	model.inference(img=img)

def simple_train_demo():
	model = MMPose()
	model.load_dataset(path=DATASET_PATH)
	model.train(epochs=10, validate=True)

def inference_from_train():
    img = 'utils/demo/000000196141.jpg'
    model = MMPose()
    model.inference(is_trained=True, pretrain_model = './checkpoints/pose_model/latest.pth',img=img)


def normal_train_demo():
	model = MMPose(backbone='HrNet32')
	model.load_dataset(path=DATASET_PATH)
	model.save_fold = "checkpoints/pose_model/"
	model.train(epochs=60, validate=True)
	model.inference(is_trained=True, pretrain_model = './checkpoints/pose_model/latest.pth',img='utils/demo/000000000785.jpg')


def continue_train_demo():
	model = MMPose(backbone='HrNet32')
	model.load_dataset(path=DATASET_PATH)
	model.save_fold = "checkpoints/pose_model/"
	model.train(epochs=15, checkpoint='./checkpoints/pose_model/latest.pth', validate=True)


if __name__ == "__main__":
	only_infer_demo()
	# simple_train_demo()
	# normal_train_demo()
	# continue_train_demo()
    # inference_from_train()