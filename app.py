

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

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# from flask import Flask, redirect, url_for, request, render_template, send_file, jsonify
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename

from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader(['./templates']))

from sanic import Sanic, response

# Any results you write to the current directory are saved as output.

# Define a flask app

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


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = DetectorConfig()
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                        config=inference_config,
                        model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
model_path = "weights/mask_rcnn_pneumonia_0010.h5"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
print("************MODEL IS LOADED****************")


# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors

app = Sanic(__name__)

app.static('pneumonia/static', './static')
@app.route('/', methods=['GET'])

def index(request):
    data = {'name': 'name'}
    template = env.get_template('index.html')
    html_content = template.render(name=data["name"])
    # Main page
    return response.html(html_content)


def predict(request, model):
    f = request.files.get('file')

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.name))
    write = open(file_path, 'wb')
    write.write(f.body)

    print("*********************************image is read! *****************************")

    ds = pydicom.read_file(file_path)
    # original image 
    image = ds.pixel_array
   
    ORIG_SIZE  = 1024
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
    print("********************************image is resized!! *****************************")
    results = model.detect([resized_image])
    r = results[0]
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (50, 50) 
    
    # fontScale 
    fontScale = 1
    
  
    color = (51,255, 51)
    mask_color = (102,178,255)

    # Line thickness of 2 px 
    thickness = 2

    mask_color = (102,178,255)
    # Line thickness of 2 px 
    thickness = 2
    
    for i in range(len(r['rois'])):
        print(image.shape)
        mask = r["masks"][:, :, i]

        image = visualize.apply_mask(resized_image, mask, color =mask_color, alpha=0.0008)
    
    image = cv2.resize(image, (1024,1024), cv2.INTER_LINEAR)

    for i in range(len(r['rois'])): 

        x1 = int(r['rois'][i][1] * resize_factor)
        y1 = int(r['rois'][i][0] * resize_factor)
        x2 = int(r['rois'][i][3] * resize_factor)
        y2 = int(r['rois'][i][2]  * resize_factor)
        cv2.rectangle(image, (x1,y1), (x2,y2), (77, 255, 9), 3, 1)
        
        x_ = x1
        y_ = y1-40
        x_1 = x1
        y_1= y1-80
        cv2.putText(image, 'Detected Pneumonia', (x_, y_), cv2.FONT_HERSHEY_SIMPLEX,
        fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.putText(image, 'confidence = {:.2f}'.format(r['scores'][i]), (x_1, y_1), cv2.FONT_HERSHEY_SIMPLEX,
        fontScale, color, thickness, cv2.LINE_AA)

        print(r['scores'][i])
        width = x2 - x1 
        height = y2 - y1 
  
    file_name = '{}_detected.png'.format(secure_filename(f.name).split(".")[0])

    return results[0]['scores'], file_name, image


# In[ ]:

@app.route('/pneumonia/predict', methods=['GET', 'POST'])
def make_preds(request):
    test_predictions, file_name, image =  predict(request, model)
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
    return response.json({
        'file_name': file_name,
        'status': status,
    })



# Callback to grab an image given a local path
@app.route('/pneumonia/get_image')
def get_image(request):
    path = request.args.get('p')
    _, ext = os.path.splitext(path)
    exists = os.path.isfile(path)
    if exists:
        return response.file(path, mime_type='image/' + ext[1:])
    

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True, access_log=False, workers=1)

# In[ ]:




