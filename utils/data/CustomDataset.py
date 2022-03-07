import mmcv
import numpy as np

from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class MyDataset(CocoDataset):
    def __init__(self, ann_file, pipeline, classes=None, data_root=None, img_prefix='', seg_prefix=None, proposal_file=None, test_mode=False, filter_empty_gt=True, file_client_args=...):
        super().__init__(ann_file, pipeline, classes, data_root, img_prefix, seg_prefix, proposal_file, test_mode, filter_empty_gt, file_client_args)
        # super(MyDataset,self)
        CLASSES = ('person', 'bicycle', 'car', 'motorcycle','plate')
    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)

        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if ann_line != '#':
                continue

            img_shape = ann_list[i + 2].split(' ')
            width = int(img_shape[0])
            height = int(img_shape[1])
            bbox_number = int(ann_list[i + 3])

            anns = ann_line.split(' ')
            bboxes = []
            labels = []
            for anns in ann_list[i + 4:i + 4 + bbox_number]:
                bboxes.append([float(ann) for ann in anns[:4]])
                labels.append(int(anns[4]))

            data_infos.append(
                dict(
                    filename=ann_list[i + 1],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                ))

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
