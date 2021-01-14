# python 3.6

import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import cv2
import face_feature
from dnnlib import tflib

from perceptual_model import PerceptualModel
from utils.visualizer import adjust_pixel_range
from utils.visualizer import save_image, load_image, resize_image

def getContextImage(image_idx):
    hair_style = {
        '1':'./aligned_images/man_dandy.png',
        '2':'./aligned_images/man_formard.png',
        '3':'./aligned_images/man_spinswallow.png',
        '4':'./aligned_images/man_asperm.png',
        '5':'./aligned_images/man_lizent.png',
        '6':'./aligned_images/man_twoblock_garma.png',
        '7':'./aligned_images/man_cropcut.png',
        '8':'./aligned_images/man_shadow.png',
        '9':'./aligned_images/man_wolf.png'
    }
    return hair_style[image_idx]

def fourChannels(img):
    print("Make target image four channels.")
    height, width, channels = img.shape
    if channels < 4:
      new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
      return new_img

    return img

def cut(img):
    # crop image
    print("Cut target image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
  
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x,y,w,h = cv2.boundingRect(cnt)
    new_img = img[y:y+h, x:x+w]
  
    return new_img

def transBg(img):
    print("Make target image background transparent.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
  
    roi, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    mask = np.zeros(img.shape, img.dtype)
  
    cv2.fillPoly(mask, roi, (255,)*img.shape[2], )
  
    masked_image = cv2.bitwise_and(img, mask)
  
    return masked_image

def cropWithWhite(img):
    print("Make target image background white.")
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    print("Loading face feature points.")
    points = face_feature.feature(img)
    # points = np.array([[[90, 90], [180, 90], [210, 110], [210, 140], [180, 210], [90, 210], [50, 140], [50, 110]]])
    # method 1 smooth region
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    # crate the white background of the same size of original image
    wbg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    # overlap the resulted cropped image on the white background
    dst = wbg + res
    cropped = dst[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    return cropped, rect

def cropWithEllipse(img):
    print("Make target image background white.")
    # create a mask
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)

    pt1 = cv2.ellipse2Poly((128, 144), (60, 80), 0, 0, 360, 5)
    cv2.drawContours(mask, [pt1], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(pt1)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    wbg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    # overlap the resulted cropped image on the white background
    dst = wbg + res
    cropped = dst[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    crop_size = 110  # default crop size.
    center_x = 60
    center_y = 80
    crop_x = center_x - crop_size // 2  # default coordinate-X
    crop_y = center_y - crop_size // 2  # default coordinate-Y
    cropped = cropped[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return cropped, rect

def createStitchedImage(context_image, cropped_image, rect):
    print("Creating Stitched Image.")

    for c in range(0, 3):
      context_image[rect[1]:rect[1] + cropped_image.shape[0], rect[0]:rect[0] + cropped_image.shape[1], c] = \
        cropped_image[:, :, c] * (cropped_image[:, :, 3] / 255.0) + \
        context_image[rect[1]:rect[1] + cropped_image.shape[0], rect[0]:rect[0] + cropped_image.shape[1], c] * (1.0 - cropped_image[:, :, 3] / 255.0)

    return context_image

def cropImage(target_image, context_image):
    mask = np.zeros(target_image.shape[0:2], dtype=np.uint8)
    # points = np.array([[[90, 90], [180, 90], [210, 110], [210, 140], [180, 210], [90, 210], [50, 140], [50, 110]]])
    points = np.array([[[70, 90], [190, 90], [190, 210], [70, 210]]])
    # points = np.array([[[70, 90], [190, 90], [190, 170], [170, 210], [90, 210], [70, 170]]])
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(target_image,target_image,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    wbg = np.ones_like(target_image, np.uint8) * 255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    # overlap the resulted cropped image on the white background
    dst = wbg + res

    context_image[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]] = (
      target_image[rect[1]:rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    )

    return context_image

def invert(model_path, _image, _wp, _latent_shape):
    print("Inverting")
    tflib.init_tf({'rnd.np_random_seed': 1000})
    with open(model_path, 'rb') as f:
        E, _, _, Gs = pickle.load(f)

    # Get input size.
    image_size = E.input_shape[2]
    assert image_size == E.input_shape[3]

    # Build graph.
    print("Inverting : Build Graph.")
    sess = tf.get_default_session()

    batch_size = 4
    input_shape = E.input_shape
    input_shape[0] = batch_size  # default batch size
    x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
    x_255 = (tf.transpose(x, [0, 2, 3, 1]) + 1) / 2 * 255

    wp = _wp
    x_rec = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
    x_rec_255 = (tf.transpose(x_rec, [0, 2, 3, 1]) + 1) / 2 * 255

    w_enc = E.get_output_for(x, phase=False)
    wp_enc = tf.reshape(w_enc, _latent_shape)
    setter = tf.assign(wp, wp_enc)

    # Settings for optimization.
    print("Inverting : Settings for Optimization.")
    perceptual_model = PerceptualModel([image_size, image_size], False)
    x_feat = perceptual_model(x_255)
    x_rec_feat = perceptual_model(x_rec_255)
    loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=[1])
    loss_pix = tf.reduce_mean(tf.square(x - x_rec), axis=[1, 2, 3])
    w_enc_new = E.get_output_for(x_rec, phase=False)
    wp_enc_new = tf.reshape(w_enc_new, _latent_shape)
    loss_enc = tf.reduce_mean(tf.square(wp - wp_enc_new), axis=[1,2])
    loss = (loss_pix +
            5e-5 * loss_feat +
            2.0 * loss_enc)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, var_list=[wp])
    tflib.init_uninitialized_vars()

    # Invert image
    print("Start Inverting.")
    num_iterations = 40
    num_results = 2
    save_interval = num_iterations // num_results

    context_images = np.zeros(input_shape, np.uint8)

    context_image = resize_image(load_image(_image), (image_size, image_size))

    # Inverting Context Image.
    context_images[0] = np.transpose(context_image, [2, 0, 1])
    context_input = context_images.astype(np.float32) / 255 * 2.0 - 1.0

    sess.run([setter], {x: context_input})
    context_output = sess.run([wp, x_rec])
    context_output[1] = adjust_pixel_range(context_output[1])
    context_image = np.transpose(context_images[0], [1, 2, 0])

    for step in tqdm(range(1, num_iterations + 1), leave=False):
        sess.run(train_op, {x: context_input})
        if step == num_iterations or step % save_interval == 0:
            context_output = sess.run([wp, x_rec])
            context_output[1] = adjust_pixel_range(context_output[1])
            if step == num_iterations: context_image = context_output[1][0]

    return context_image

def encode(_target_image, _context_image, _output_dir):
    gpu_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(_target_image)
    assert os.path.exists('./static/'+_target_image)
    _output_dir = _output_dir[:-4]
    output_dir = './static/' + _output_dir

    tflib.init_tf({'rnd.np_random_seed': 1000})
    model_path = './styleganinv_face_256.pkl'
    with open(model_path, 'rb') as f:
        E, _, _, Gs = pickle.load(f)

    # Get input size.
    image_size = E.input_shape[2]
    assert image_size == E.input_shape[3]

    crop_size = 110  # default crop size.
    center_x = 125
    center_y = 145
    crop_x = center_x - crop_size // 2  # default coordinate-X
    crop_y = center_y - crop_size // 2  # default coordinate-Y

    mask = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
    mask[:, crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :] = 1.0

    # Build graph.
    sess = tf.get_default_session()

    batch_size = 4
    input_shape = E.input_shape
    input_shape[0] = batch_size  # default batch size
    x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
    x_mask = (tf.transpose(x, [0, 2, 3, 1]) + 1) * mask - 1
    x_mask_255 = (x_mask + 1) / 2 * 255

    latent_shape = Gs.components.synthesis.input_shape
    latent_shape[0] = batch_size  # default batch size
    wp = tf.get_variable(shape=latent_shape, name='latent_code')
    x_rec = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
    x_rec_mask = (tf.transpose(x_rec, [0, 2, 3, 1]) + 1) * mask - 1
    x_rec_mask_255 = (x_rec_mask + 1) / 2 * 255

    w_enc = E.get_output_for(x, phase=False)
    wp_enc = tf.reshape(w_enc, latent_shape)
    setter = tf.assign(wp, wp_enc)

    # Settings for optimization.
    print("Diffusion : Settings for Optimization.")
    perceptual_model = PerceptualModel([image_size, image_size], False)
    x_feat = perceptual_model(x_mask_255)
    x_rec_feat = perceptual_model(x_rec_mask_255)
    loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=[1])
    loss_pix = tf.reduce_mean(tf.square(x_mask - x_rec_mask), axis=[1, 2, 3])

    loss_weight_feat = 5e-5
    learning_rate = 0.01
    loss = loss_pix + loss_weight_feat * loss_feat  # default The perceptual loss scale for optimization. (default 5e-5)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, var_list=[wp])
    tflib.init_uninitialized_vars()

    # Invert image
    num_iterations = 100
    num_results = 5
    save_interval = num_iterations // num_results

    images = np.zeros(input_shape, np.uint8)
   
    print("Load target image.")
    _target_image = './static/' + _target_image
    target_image = resize_image(load_image(_target_image), (image_size, image_size))
    save_image('./' + output_dir + '_tar.png', target_image)

    print("Load context image.")
    context_image = getContextImage(_context_image)
    context_image = resize_image(load_image(context_image), (image_size, image_size))
    save_image('./' + output_dir + '_cont.png', context_image)

    # Inverting Context Image.
    # context_image = invert(model_path, getContextImage(_context_image), wp, latent_shape)
    save_image('./' + output_dir + '_cont_inv.png', context_image)

    # Create Stitched Image
    # context_image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size] = (
    #     target_image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    # )
    # context_image[crop_y:crop_y + 170, crop_x - 70:crop_x + crop_size + 190] = (
    #     target_image[crop_y:crop_y + 170, crop_x - 70:crop_x + crop_size + 190]
    # )
    print("Cropping Image...")
    # context_image = cropImage(target_image, context_image)

    target_image, rect = cropWithWhite(target_image)
    target_image = fourChannels(target_image)
    target_image = cut(target_image)
    target_image = transBg(target_image)

    context_image = createStitchedImage(context_image, target_image, rect)
    save_image('./' + output_dir + '_sti.png', context_image)
    images[0] = np.transpose(context_image, [2, 0, 1])

    input = images.astype(np.float32) / 255 * 2.0 - 1.0

    # Run encoder
    print("Start Diffusion.")
    sess.run([setter], {x: input})
    output = sess.run([wp, x_rec])
    output[1] = adjust_pixel_range(output[1])

    col_idx = 4
    for step in tqdm(range(1, num_iterations + 1), leave=False):
        sess.run(train_op, {x: input})
        if step == num_iterations or step % save_interval == 0:
            output = sess.run([wp, x_rec])
            output[1] = adjust_pixel_range(output[1])
            if step == num_iterations:
                save_image(f'{output_dir}.png', output[1][0])
            col_idx += 1
    exit()

if __name__ == '__main__':
    #args = parse_args()
    output_dir = './results/diffusion6'
    #encode(args.target_image, args.context_image, output_dir)
