import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from utils import *

def main(argv):

    if isinstance(FLAGS.image,str):
        image = cv2.imread(FLAGS.image,0)
    plt.imshow(image, cmap='gray')
    plt.title('Cropped image')
    plt.pause(1)
    # Rotation of the image
    r_img = rotate_img(image)
    plt.imshow(r_img, cmap='gray')
    plt.title('Cropped image')
    plt.pause(1)
    # Expansion of the image
    exp_img = expansion_img(r_img)
    plt.imshow(exp_img, cmap='gray')
    plt.title('Cropped image')
    plt.pause(1)
    # Egalisation of the image
    eg_img = egalisation_img(exp_img)
    plt.imshow(eg_img, cmap='gray')
    plt.title('Cropped image')
    plt.pause(1)
    # Detection of the color of the digits of the license plate
    # If they are black we inverse th grays of the image
    b_img = black_or_white(eg_img)
    plt.imshow(b_img, cmap='gray')
    plt.title('black or white image')
    plt.pause(1)
    # Vertical cut
    cut_img = cropping_border(b_img)
    plt.imshow(cut_img, cmap='gray')
    plt.title('Cropped image')
    plt.pause(1)
    # Binarization
    b_img = binarization(cut_img)
    plt.imshow(b_img, cmap='gray')
    plt.title('Cropped image')
    plt.pause(1)
    # Vertical crop
    v_crop_img = vertical_crop(b_img)
    plt.imshow(v_crop_img, cmap='gray')
    plt.title('Cropped image')
    plt.pause(1)
    # Horizontal crop
    h_crop_img = horizontal_crop(v_crop_img,4)
    plt.imshow(h_crop_img, cmap='gray')
    plt.title('Cropped image')
    plt.pause(1)
    # Segmentation
    caracter_list_image = segmentation(h_crop_img)
    # Remove the noises from the cropped results
    caracter_list_image = remove_noises(caracter_list_image)
    characters = []
    # The digits may not appear be shown in the rigth order but it 
    # doesn't matter because the digits can be sorted by their
    # position from the right to the left
    for i, character in enumerate(caracter_list_image):
        character = crop_char_and_save(character)
        characters.append(character)
        plt.imshow(character, cmap='gray')
        plt.title('digit'.format(i))
        plt.pause(1)
    plt.close()
    
    return characters

if __name__ == "__main__":
    
    FLAGS = flags.FLAGS
    flags.DEFINE_string('image','./plate_test.jpg','The path of the image you want to segment')

    app.run(main)