import base64

def nuRLE2RLE(mask: dict) -> dict:
    '''
    :param mask: mask for segmentation in nuimage with RLE not able to be decoded
    :return: mask for segmentation in coco format
    '''

    new_mask = (mask.copy())
    new_mask["counts"]=base64.b64decode(mask["counts"])
    #decode to make it RLE
    new_mask['counts'] = new_mask["counts"].decode('utf-8')
    # bytes are not json serializable
    #so make it str
    return new_mask
