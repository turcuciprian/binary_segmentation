# training the model -- FULL CODE --
from fastai.vision.all import *

path = Path('data')

lbl_names = get_image_files('./data/GT_png')
fnames = get_image_files('./data/images')

img_fn = fnames[2]

im = PILImage.create('./data/images/00007.png')
msk = PILMask.create('data/GT_png/00007_mask.png')


def get_msk(fn, p2c):
    "Grab a mask from a `filename` and adjust the pixels based on `pix2class`"
    fn = path/''/f'GT_png'/f'{fn.stem}_mask.png'
    msk = np.array(PILMask.create(fn))
    mx = np.max(msk)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)


def n_codes(fnames, is_partial=True):
    "Gather the codes from a list of `fnames`"
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


p2c = n_codes(lbl_names)

codes = ['Background', 'Wheel']


def get_y(o): return get_msk(o, p2c)


binary = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_y,
                   item_tfms=Resize(224),
                   batch_tfms=[Normalize.from_stats(*imagenet_stats)])

dls = binary.dataloaders('data/images', bs=2)

dls.show_batch(cmap='Blues', vmin=0, vmax=1)

learn = unet_learner(dls, resnet34)

learn.fit(10)

preds = learn.get_preds()

p = preds[0][0]