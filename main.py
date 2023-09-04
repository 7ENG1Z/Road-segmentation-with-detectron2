from Detector import *

detector = Detector()
# detector.get_trained_mode(cfg_save_path=,MODEL_DIR=)
# detector.show_image("images\\n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915374762465.jpg")

detector.save_mask("testdataset",'maskofroad.json')#the name of json file end with .json
