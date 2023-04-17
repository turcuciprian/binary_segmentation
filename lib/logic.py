# training the model -- FULL CODE --
from fastai.vision.all import *
import os
import cv2


def get_msk(fn, p2c):
    # "Grab a mask from a `filename` and adjust the pixels based on `pix2class`"
    fn = path/''/f'GT_png'/f'{fn.stem}_mask.png'
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
    
lbl_names = sorted(get_image_files(string_path+'/GT_png'))
fnames = sorted(get_image_files(string_path+'/images'))

p2c = n_codes(lbl_names)

def main_logic():
    img_fn = fnames[2]
    im = PILImage.create(fnames[0])
    msk = PILMask.create(lbl_names[0])
    codes = ['Background', 'Wheel']
    binary = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_y,
                   #item_tfms=Resize(224),
                   batch_tfms=[Normalize.from_stats(*imagenet_stats)])
                   

    dls = binary.dataloaders(string_path+'/images', bs=2)

    learn = unet_learner(dls, resnet34)
    return learn

def get_predict_image(file,learn):
    dl = learn.dls.test_dl(file)
    preds = learn.get_preds(dl=dl)
    pred_1 = preds[0][0]
    pred_arx = pred_1.argmax(dim=0)
    pred_arx = pred_arx.numpy()
    rescaled = (255.0 / pred_arx.max() *
                (pred_arx - pred_arx.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    return im

def overlay_mask(image, mask):
    # Convert the binary mask to a color mask (3 channels)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask_color)

    # Add the mask to the image as a transparent overlay
    overlay = cv2.addWeighted(image, 0.5, masked_image, 0.9, 0)

    return overlay