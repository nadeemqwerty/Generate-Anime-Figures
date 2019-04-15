from network import gen, disc
from data_utils import load_batch, denorm_img, combine_images

from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np
import datetime
from PIL import Image
import argparse

def train(epochs = 1, batch_size = 64, sample_interval = 100):
    start_time = datetime.datetime.now()

    disc = disc()
    gen = gen()

    disc.trainable = False

    adam = Adam(lr=0.00015, beta_1=0.5)

    gan = Sequential()
    gan.add(gen)
    gan.add(disc)

    gan.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    gan.summary()

    images = np.load("")
    labels = np.load("")

    for epoch in range(epochs):
        for i_batch, (image_batch, label_batch) in enumerate(load_batch(images, labels, batch_size = batch_size)):

            fake_image = gen.predict(label_batch)

            real_fake = np.concatenate((image_batch, fake_image))

            disc_label_real = np.ones(batch_size) - np.random.random(batch_size)*0.1
            disc_label_fake = np.random.random(batch_size)*0.1

            disc_label = np.concatenate((disc_label_real,disc_label_fake))

            disc.trainable = True
            gen.trainable = False

            loss_disc = disc.train_on_batch(real_fake, disc_label)

            loss_d.append(loss_disc)

            gen.trainable = True
            disc.trainable = False

            loss_gen = gan.train_on_batch(label_batch,disc_label_real)

            loss_g.append(loss_gen)

            elapsed_time = datetime.datetime.now() - start_time
            
            print ("[%d] [%d/%d] time: %s, [d_loss: %f, g_loss: %f]" % (epoch, i_batch,
                                                                        n_batches,
                                                                        elapsed_time,
                                                                        loss_d, loss_gen))
            if i_batch % sample_interval == 0:
                disc.save("model/disc.h5")
                gen.save("model/gen.h5")

                image = denorm_img(fake_image)
                image = combine_images(image)
                Image.fromarray(image.astype(np.uint8)).save("train_img/"+
                    str(epoch)+"_"+str(i_batch)+".png")
