import numpy
import matplotlib.pyplot as plt
import math

def denorm_img(img):
    img = (img + 1) * 127.5
    return img

def load_batch(images,label, batch_size = 64, label_with_noise = True):
    n_batches = int(len(data.shape[0]/batch_size))

    for i in range(n_batches-1):
        image_batch = data[i*batch_size:(i+1)*batch_size]

        if label_with_noise:
            label_batch = label[i*batch_size:(i+1)*batch_size]

        else :

        image_batch = (image_batch/127.5) - 1.

        yield image_batch, label_batch

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] =             img[:, :, 0]
    return image
