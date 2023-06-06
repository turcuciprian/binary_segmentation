# training the model -- FULL CODE --
from fastai.vision.all import *
import os
import cv2


def get_msk(fn, p2c):
    # "Grab a mask from a `filename` and adjust the pixels based on `pix2class`"
    fn = path/''/f'masks'/f'{fn.stem}.png'
    msk = np.array(PILMask.create(fn))
    mx = np.max(msk)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)

def n_codes(fnames, is_partial=True):
    # "Gather the codes from a list of `fnames`"
    vals = set()
    if is_partial:
        random.shuffle(fnames)
        fnames = fnames[:len(fnames)]
    for fname in fnames:
        msk = np.array(PILMask.create(fname))
        for val in np.unique(msk):
            if val not in vals:
                vals.add(val)
    vals = list(vals)
    p2c = dict()
    for i, val in enumerate(vals):
        p2c[i] = vals[i]
    return p2c





def get_y(o): return get_msk(o, p2c)

string_path = './data/'
path = Path(string_path)
    
lbl_names = sorted(get_image_files(string_path+'/masks'))
fnames = sorted(get_image_files(string_path+'/images'))

p2c = n_codes(lbl_names)

def main_logic(training=True):
    img_fn = fnames[2]
    im = PILImage.create(fnames[0])
    msk = PILMask.create(lbl_names[0])
    codes = ['Background', 'Wheel']
    binary = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_y,
                   item_tfms= Resize(500) if training == True else None, # 224 original - for inference, comment this line (leave for training)
                   batch_tfms=[Normalize.from_stats(*imagenet_stats)])
                   

    dls = binary.dataloaders(string_path+'/images', bs=1)

    learn = unet_learner(dls, resnet34)
    return learn
