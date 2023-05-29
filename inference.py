# INFERENCE - load the pretrained model and use it with images that where not used for training (images from google images, from phone camera..etc)

from IPython.display import Image
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
# from PIL import Image

sys.path.append('lib')
from logic import *
from general import *
learn = main_logic()
# learn.load('model_v0.1')
# learn.load('Neuromania-v.0.5')
learn.load('model_v0.2')
# learn.load('neuromania')

# input and output directory for testing
input_folder = './test/input/'
output_folder = './test/output/'
print('resizing the images...')
resize_images(input_folder, 800)
print('creating the masks...')
# file and mask processing
for filename in os.listdir(input_folder):
        if os.path.isfile(os.path.join(input_folder, filename)):
            image_path=input_folder+filename
            first_mask = get_predict_image(image_path,learn)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # Convert the PIL Image object to a NumPy array
            img_np = np.asarray(first_mask)
            # Convert the NumPy array to a cv2.imread() format
            overlayed_mask = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            # Overlay the mask onto the image
            overlay = overlay_mask(image, overlayed_mask)
            # Convert the NumPy array back to an image
            image = Image.fromarray(overlay)

            # Save the image to disk
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_mask{ext}"
        image.save(os.path.join(output_folder, new_filename))
