import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
ROOT_DIR = ''
import glob
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import warnings 
warnings.filterwarnings("ignore")


def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations 



class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01

    STEPS_PER_EPOCH = 200

config = DetectorConfig()

# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors

    import warnings 


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
model_path = "Mask_RCNN/weights/mask_rcnn_pneumonia_0010.h5"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


def predict(ORIG_SIZE = 1024): 
    #image_id = random.choice(test_image_fps)
    image_id = 'uploads/0cf2d6d5-6124-47f7-94ed-78f1aefea44b.dcm'
    ds = pydicom.read_file(image_id)
    
    # original image 
    image = ds.pixel_array
    
    # assume square image 
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1) 
    resized_image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)


    results = model.detect([resized_image])
    r = results[0]
    for bbox in r['rois']: 
        print(bbox)
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2]  * resize_factor)
        cv2.rectangle(image, (x1,y1), (x2,y2), (77, 255, 9), 3, 1)
        width = x2 - x1 
        height = y2 - y1 

        #plt.figure() 
    #plt.imshow(image, cmap=plt.cm.gist_gray)
    
    file_name = '{}_detected.png'.format(image_id.split(".dicom")[0])

    return results[0]['scores'], file_name, image



def make_preds():
    test_predictions, file_name, image =  predict(ORIG_SIZE = 1024)
    status = 'detected.'
    if test_predictions[0] > 0.4:

        status = 'detected.'
        print(f'Pneumonia positive detected with confidence of: {test_predictions[0]}')
        cv2.imwrite(file_name, image)
        plt.figure() 
        plt.imshow(image, cmap=plt.cm.gist_gray)
        plt.axis('off')
    else:
        status = 'not detected.'
        print('Pneumonia not detected')
        cv2.imwrite(file_name, image)
        plt.figure() 
        plt.imshow(image, cmap=plt.cm.gist_gray)
        plt.axis('off')
    return 
make_preds()