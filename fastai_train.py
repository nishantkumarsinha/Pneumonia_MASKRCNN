from fastai.vision import *
from fastai.basics import *
from fastai.metrics import error_rate
import pydicom
import imageio
import PIL
import json, pdb


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.imports import *

from matplotlib.patches import Rectangle
import os
import seaborn as sns
import pydicom


def open_dcm_image(fn:PathOrStr,convert_mode:str='RGB',after_open:Callable=None)->Image:
    "Return `Image` object created from image in file `fn`."
    array = pydicom.dcmread(fn).pixel_array
    x = PIL.Image.fromarray(array).convert('RGB')
    return Image(pil2tensor(x,np.float32).div_(255))



vision.data.open_image = open_dcm_image
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0)
data = ImageDataBunch.from_csv(path,folder='stage_2_train_images',csv_labels='stage_2_train_labels.csv',ds_tfms=tfms,fn_col='patientId',label_col='Target',suffix='.dcm',seed=47,size=224, bs = 128)

learner = load_learner(Path('model'), 'chexnet_NIH.pkl')
model = learner.model
def create_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                concat_pool:bool=True, bn_final:bool=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

custom_head2=create_head(nf = 2048, nc = 2)
precision=Precision()
recall=Recall()
AUC=AUROC()
model_ch1 = nn.Sequential(model[:-1],custom_head2)
learn = Learner(data, model_ch1, metrics=(accuracy,precision,recall,AUC))
learn.fit_one_cycle(10)
