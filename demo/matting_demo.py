from MMEdu.Mating_Edu import MMMat

DATASET_PATH = '/remote-home/congpsh/mmlab/mmediting/demo/data/alphamatting'
MERGE_PATH = '/remote-home/congpsh/mmlab/OpenMMLab-Edu/mmediting/tests/data/merged/GT05.jpg'
TRIMAP_PATH = '/remote-home/congpsh/mmlab/OpenMMLab-Edu/mmediting/tests/data/trimap/GT05.png'


def only_infer_demo():
	merged_path = MERGE_PATH
	trimap_path = TRIMAP_PATH
	model = MMMat(backbone='IndexNet')
	model.inference(merged_path = merged_path, trimap_path= trimap_path)

def simple_train_demo():
	model = MMMat()
	model.load_dataset(path=DATASET_PATH)
	model.train()

def inference_from_train():
    merged_path = MERGE_PATH
    trimap_path = TRIMAP_PATH
    # img = 'dataset/coco_tiny/images/'
    model = MMMat(backbone='IndexNet')
    model.inference(is_trained=True, pretrain_model = 'checkpoints/edit_model/latest.pth',merged_path = merged_path, trimap_path= trimap_path)

def normal_train_demo():
	merged_path = MERGE_PATH
	trimap_path = TRIMAP_PATH
	model = MMMat(backbone='IndexNet')
	model.load_dataset(path=DATASET_PATH)
	model.save_fold = "checkpoints/edit_model"
	model.train(validate=True,epochs=5)
	model.inference(is_trained=True, pretrain_model = 'checkpoints/edit_model/latest.pth',merged_path = merged_path, trimap_path= trimap_path)


def continue_train_demo():
    	
	model = MMMat(backbone='IndexNet')
	model.load_dataset(path=DATASET_PATH)
	model.save_fold = "checkpoints/edit_model"
	model.train(epochs=5,checkpoint='checkpoints/edit_model/latest.pth', validate=True)


if __name__ == "__main__":
	# only_infer_demo()
	simple_train_demo()
	# normal_train_demo()
	# continue_train_demo()
    # inference_from_train()