import os
import numpy as np
from PIL import ImageEnhance
from PIL import Image
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage import transform
from skimage import io
import matplotlib

def rotate(img, rad_angle):
    afine_tf = transform.AffineTransform(rotation=rad_angle)
    rotated_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return rotated_img

def translate(img, trans_x, trans_y):
    afine_tf = transform.AffineTransform(translation=(trans_x, trans_y))
    translated_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return translated_img

def scale(img, scale_1, scale_2):
    afine_tf = transform.AffineTransform(scale=(scale_1, scale_2))
    scaled_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return scaled_img

def shear(img, value):
    afine_tf = transform.AffineTransform(shear=value)
    sheared_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return sheared_img
    
def blur(img, sigma):
    is_colour = len(img.shape)==3
    blur_img = np.uint8(rescale_intensity(gaussian(img, sigma=sigma, multichannel=is_colour,preserve_range=True),out_range=(0,255)))
    return blur_img

def change_brightness(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Brightness(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_color(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Color(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_contrast(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Contrast(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_sharpness(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Sharpness(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def normalize(img):
    norm_img = np.float32(img / 255.0)
    return norm_img

def denormlize(img):
    denorm_img = np.uint8(img * 255.0)
    return denorm_img

def store_data(id, data):
    if not os.path.isdir('./content/test_images'):
        os.mkdir('./content/test_images')
    matplotlib.image.imsave("./content/test_images/id_{}.png".format(id), data)