from keras.layers import Conv2D, Conv2DTranspose, Dropout, Dense, BatchNormalization
from keras.layers import MaxPooling2D, Reshape, Activation, Flatten, GaussianNoise, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model,Sequential
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop

def gen(shape = (512+32,)):
    kernel_init = 'glorot_uniform'

    inp = Input(shape = shape)

    x = Dense(64*16*16)(inp)
    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)

    x = Reshape((16,16,64))(x)

    x = Conv2DTranspose(filters =256 , kernel_size = (3,3), strides = (1,1), padding = "same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(filters =128 , kernel_size = (3,3), strides = (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x =GaussianNoise(0.001)(x)

    x = Conv2DTranspose(filters =64 , kernel_size = (3,3), strides = (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x =GaussianNoise(0.001)(x)

    x = Conv2D(filters =64 , kernel_size = (3,3), strides = (1,1), padding = "same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # x =GaussianNoise(0.001)(x)

    x = Conv2DTranspose(filters =3 , kernel_size = (3,3), strides = (2,2), padding = "same")(x)
    x = Activation("tanh")(x)

    model = Model(inp,x)
    adam = Adam(lr=0.00015, beta_1=0.5)
    model.compile(loss = "binary_crossentropy",optimizer = adam, metrics=['accuracy'])
    print("Generator Model Summary")
    model.summary()
    return model

def disc(shape = (128,128,3)):

    inp = Input(shape = shape)

    x = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same")(inp)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters = 128, kernel_size = (3,3), strides = (2,2), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters = 256, kernel_size = (3,3), strides = (2,2), padding = "same")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters = 512, kernel_size = (3,3), strides = (2,2), padding = "same")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)

    x = Dense(1, activation = 'sigmoid')(x)

    model = Model(inp,x)
    adam = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss = 'binary_crossentropy',optimizer=adam, metrics = ['accuracy'])

    print('Discriminator Model Summary')
    model.summary()

    return model

g = gen()
d = disc()
