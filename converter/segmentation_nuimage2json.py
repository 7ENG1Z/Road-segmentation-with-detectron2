from nuimages import NuImages
import json
import numpy as np
from utils import*
from pycocotools import mask

nuim = NuImages(dataroot='./data/sets/nuimages', version='v1.0-mini', verbose=True, lazy=False)
# print(nuim.category[0]) #test if the nuimage is loaded

#initialize the structure of coco format

surface_anno= {
    'images': [],
    'annotations': [],
    'categories': [
        {"name":"road",
         "id":1
        }
    ]
}

for i in range(len(nuim.sample)):
    sample_data_token = nuim.sample[i]["key_camera_token"]

    #get the corresponding sam_ple data and surface_ann
    sample_data = nuim.get('sample_data', sample_data_token)
    image = {
            'id':sample_data_token ,
            'file_name': sample_data['filename'],
            'width':  sample_data['width'],
            'height':  sample_data['height']
        }
    surface_anno['images'].append(image)

surface_anns = [o for o in nuim.surface_ann if o['category_token'] == "a86329ee68a0411fb426dcad3b21452f"]
# print(len(surface_anns))
for j in range(len(surface_anns)):
    annotation={
                'id':surface_anns[j]['token'],
                'category_id':1,
                'segmentation': nuRLE2RLE(surface_anns[j]['mask']),
                'bbox':mask.toBbox(nuRLE2RLE(surface_anns[j]['mask'])).tolist(),
                'image_id':surface_anns[j]['sample_data_token']
        }
    surface_anno['annotations'].append(annotation)

# save as json file
with open('road_annotation.json', 'w') as f:
    json.dump(surface_anno, f)