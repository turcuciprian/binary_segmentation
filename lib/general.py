from PIL import Image, ImageDraw
import cv2
from fastai.vision.all import *


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
    mask_color=cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask_color)

    # Add the mask to the image as a transparent overlay
    overlay = cv2.addWeighted(image, 0.3, masked_image, 1, 0)
    
    # Convert both images to the same color space (in this case, BGR)

    return overlay


def highlight_areas(pilImage):
    # Open the image and convert it to RGBA mode
    with pilImage.convert("RGBA") as im:
        # Get the pixel data
        data = im.load()
        width, height = im.size

        # Create a new image to draw on
        out_im = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(out_im)

        # Keep track of which pixels have been visited
        visited = set()

        # Find the white areas and draw a red rectangle around each area
        for x in range(width):
            for y in range(height):
                # If we haven't visited this pixel and it's white, it's the start of a new area
                if (x, y) not in visited and data[x, y][:3] == (255, 255, 255):
                    # Find the coordinates of the bounding box for this area
                    left, top = x, y
                    right, bottom = x, y
                    stack = [(x, y)]

                    while stack:
                        x, y = stack.pop()
                        if (x, y) not in visited and data[x, y][:3] == (255, 255, 255):
                            visited.add((x, y))
                            left = min(left, x)
                            top = min(top, y)
                            right = max(right, x)
                            bottom = max(bottom, y)
                            stack.extend([(x+1, y), (x, y+1), (x-1, y), (x, y-1)])

                    # If the bounding box is at least 2x2, draw a rectangle around it
                    if right - left > 3 and bottom - top > 3:
                        draw.rectangle((left, top, right, bottom), outline="red", width=1)

        # Return the new image
        return out_im
def resize_images(directory, width):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            with Image.open(os.path.join(directory, filename)) as img:
                original_width, original_height = img.size
                if(original_width>width):
                    height = int((width / float(img.size[0])) * img.size[1])
                    resized_img = img.resize((width, height))
                    resized_img.save(os.path.join(directory, filename))
                    
def is_image(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))