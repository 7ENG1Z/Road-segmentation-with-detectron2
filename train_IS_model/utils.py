from  detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode,Visualizer
import cv2
import random
import matplotlib.pyplot as plt


def plot_samples(dataset_name,n=1):
    #can be applied to image with instance coco annotation

    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata =MetadataCatalog.get(dataset_name)
    # dataset_custom_metadata =MetadataCatalog.get(dataset_name).set(thing_classes=["road"])

    for s in random.sample(dataset_custom,n):
        img=cv2.imread(s['file_name'])
        v=Visualizer(img[:,:,::-1],metadata=dataset_custom_metadata)
        v=v.draw_dataset_dict(s)

        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image()[:,:,:])
        plt.show()


def get_train_config(config_file_path, checkpoint_url,train_dataset_name, num_classes,device,output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN =(train_dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.INPUT.MASK_FORMAT = "bitmask" # we use counts to describe mask instead of "polygon"
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real "batch size"
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []  # do not decay learning rate


    # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE= device
    cfg.OUTPUT_DIR = output_dir

    return cfg

def on_image(image_path, predictor):
    im = cv2.imread(image_path)
    output = predictor(im)
    # print(output)
    v = Visualizer(im[:,:,::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(output['instances'].to('cpu'))

    plt.figure(figsize=(14,10))
    plt.imshow(v.get_image())
    plt.show()

