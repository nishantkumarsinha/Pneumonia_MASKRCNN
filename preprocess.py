import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import pydicom
import os
from os import listdir
from os.path import isfile, join
from skimage import measure
from skimage.transform import resize
import csv
import random



def parse_data(df, test = False):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    if not test:
      for n, row in df.iterrows():
          # --- Initialize patient entry into parsed 
          pid = row['patientId']
          if pid not in parsed:
              parsed[pid] = {
                  'dicom': 'data/stage_2_train_images/%s.dcm' % pid,
                  'label': row['Target'],
                  'boxes': []}

          # --- Add box if opacity is present
          if parsed[pid]['label'] == 1:
              parsed[pid]['boxes'].append(extract_box(row))
    else:
      for n, row in df.iterrows():
          # --- Initialize patient entry into parsed 
          pid = row['patientId']
          if pid not in parsed:
              parsed[pid] = {
                  'dicom': 'data/stage_2_test_images/%s.dcm' % pid,
                  'label': row['Target'],
                  'boxes': []}

          # --- Add box if opacity is present
          if parsed[pid]['label'] == 1:
              parsed[pid]['boxes'].append(extract_box(row))

    return parsed


def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        #rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = [255, 251, 204] # Just use yellow
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=15)

    plt.imshow(im, cmap=plt.cm.gist_gray)
    plt.axis('off')

def overlay_box(im, box, rgb, stroke=2):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
