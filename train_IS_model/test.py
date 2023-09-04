from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

#load config and weight
cfg_save_path = 'IS_cfg.pickle'
with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = 'nuimages/test/1.jpg' #TBA
on_image(image_path, predictor)