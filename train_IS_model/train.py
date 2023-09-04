from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle
from utils import *


config_file_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# config_file_path="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
# checkpoint_url="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
output_dir ='./output/InstanceSegmentation'
num_classes=1

device='cuda'

train_dataset_name ='nuimage_train'
train_images_path='nuimages'
train_json_annot_path='road_annotation.json'


#config save path
cfg_save_path = 'IS_cfg.pickle'

#################################################
register_coco_instances(name=train_dataset_name,metadata={},json_file=train_json_annot_path,
                        image_root=train_images_path)

# plot_samples(dataset_name=train_dataset_name,n=5)

######################
def main():
    cfg = get_train_config(config_file_path, checkpoint_url, train_dataset_name, num_classes, device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    # train and save the model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume= False)
    trainer.train()

if __name__ == '__main__':
    main()