# import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2 import model_zoo
import cv2
import numpy as np
import json
from pycocotools import mask
import os
import pickle
import matplotlib.pyplot as plt
class Detector:
    def __init__(self):
        self.cfg=get_cfg()

        # load model config and pretrained model
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        # self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        # self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =1
        self.cfg.MODEL.DEVICE = "cuda"
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.predictor=DefaultPredictor(self.cfg)

    def get_trained_mode(self,cfg_save_path,MODEL_DIR):
        with open(cfg_save_path, 'rb') as f:
            self.cfg = pickle.load(f)
        self.cfg.MODEL.WEIGHTS = os.path.join(MODEL_DIR, 'model_final.pth')
        self.predictor = DefaultPredictor(self.cfg)

    def show_image(self,imagePath):
        '''
        show the result of segmentation on image
        '''
        #read image
        image = cv2.imread(imagePath)
        #get prediction
        predictions,segmentInfo=self.predictor(image)["panoptic_seg"]
        #show the picture
        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), list(filter(lambda x: x['category_id'] == 21, segmentInfo)), area_threshold=.1)
        window_name_output = 'output'
        cv2.namedWindow(window_name_output, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name_output,output.get_image()[:,:,::-1])
        cv2.waitKey(0)

    def save_mask(self,testfolder_path,json_file_name):
        '''
        make predictions of road on the images in the file folder'
        and save the mask of road in coco format!!
        '''

        # initialize the structure of coco format
        surface_anno = {
            'images': [],
            'annotations': [],
            'categories': [
                {"name": "road",
                 "id": 21
                 }
            ]
        }
        id_image=0
        id_anno=0
        # the folder path
        # get the file list in folder
        image_files = [f for f in os.listdir(testfolder_path) if os.path.isfile(os.path.join(testfolder_path, f))]

        for image_file in image_files:
            #get the image path
            image_path = os.path.join(testfolder_path, image_file)
            # read image
            image = cv2.imread(image_path)
            # get prediction
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            m = predictions.to("cpu").numpy().astype(np.uint8)
            #generate the mask of the road
            ann = list(filter(lambda x: x['category_id'] == 21, segmentInfo))#for detectron2, the ID of road is 21. Change this for different ID
            for i in range(len(m)):
                for j in range(len(m[0])):
                    if m[i][j] != ann[0]['id']:
                        m[i][j] = 0
            #make the mask RLE
            RLE =mask.encode(np.asfortranarray(m))
            RLE["counts"]=RLE["counts"].decode('utf-8')
            # print(RLE)
            # decoded_mask = mask.decode(RLE)
            # plt.imshow(decoded_mask)
            # plt.show()

            #write in json
            images = {
                'id': id_image,
                'file_name':image_path ,
                'width': image.shape[1],
                'height':image.shape[0]
            }
            surface_anno['images'].append(images)

            annotations = {
                'id': id_anno,
                'category_id': 21,
                # 'segmentation': nuRLE2RLE(surface_anns[j]['mask']),
                'segmentation': RLE,
                'bbox': mask.toBbox(RLE).tolist(),
                'image_id': id_image
            }
            surface_anno['annotations'].append(annotations)
            id_anno=id_anno+1
            id_image=id_image+1

        # save as json file
        with open(json_file_name, 'w') as f:
            json.dump(surface_anno, f)






