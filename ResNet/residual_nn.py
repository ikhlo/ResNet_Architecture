from keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D, Add, AveragePooling2D


CHANNEL_AXIS = 1 if K.image_data_format() == "channels_first" else -1

def res_identity_block(x, no_filters_list, n_strides=1):

  x_init = x
  f1, f2 = no_filters_list

  x = Conv2D(filters = f1, kernel_size=1, strides=n_strides, kernel_initializer='he_normal', use_bias=False)(x)
  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)

  x = Conv2D(filters = f1, kernel_size=3, padding="same", strides=n_strides, kernel_initializer='he_normal', use_bias=False)(x)
  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)

  x = Conv2D(filters = f2, kernel_size=1, strides=n_strides, kernel_initializer='he_normal', use_bias=False)(x)
  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Add()([x, x_init])
  x = Activation('relu')(x)

  return x


def res_conv_block(x, no_filters_list, n_strides):

  x_init = x
  f1, f2 = no_filters_list

  x = Conv2D(filters = f1, kernel_size=1, strides=n_strides, kernel_initializer='he_normal', use_bias=False)(x)
  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)

  x = Conv2D(filters = f1, kernel_size=3, padding="same", strides=1, kernel_initializer='he_normal', use_bias=False)(x)
  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)

  x = Conv2D(filters = f2, kernel_size=1, strides=1, kernel_initializer='he_normal', use_bias=False)(x)
  x = BatchNormalization(axis=CHANNEL_AXIS)(x)

  x_init = Conv2D(filters = f2, kernel_size=1, strides=n_strides, kernel_initializer='he_normal', use_bias=False)(x_init)
  x_init = BatchNormalization(axis=CHANNEL_AXIS)(x_init)

  x = Add()([x, x_init])
  x = Activation('relu')(x)

  return x


def resnet_model(input_shape:tuple, nb_of_classes:int):

  model_input = Input(shape=input_shape)
  x = ZeroPadding2D(padding=(3, 3))(model_input)

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = res_conv_block(x, no_filters_list=(64, 256), n_strides= 1)
  x = res_identity_block(x, no_filters_list=(64, 256))
  x = res_identity_block(x, no_filters_list=(64, 256))

  x = res_conv_block(x, no_filters_list=(128, 512), n_strides= 2)
  x = res_identity_block(x, no_filters_list=(128, 512))
  x = res_identity_block(x, no_filters_list=(128, 512))
  x = res_identity_block(x, no_filters_list=(128, 512))

  x = res_conv_block(x, no_filters_list=(256, 1024), n_strides= 2)
  x = res_identity_block(x, no_filters_list=(256, 1024))
  x = res_identity_block(x, no_filters_list=(256, 1024))
  x = res_identity_block(x, no_filters_list=(256, 1024))
  x = res_identity_block(x, no_filters_list=(256, 1024))
  x = res_identity_block(x, no_filters_list=(256, 1024))

  x = res_conv_block(x, no_filters_list=(512, 2048), n_strides= 2)
  x = res_identity_block(x, no_filters_list=(512, 2048))
  x = res_identity_block(x, no_filters_list=(512, 2048))

  x = AveragePooling2D(pool_size=(2,2))(x)
  x = Flatten()(x)
  x = Dense(nb_of_classes, activation="softmax")(x)

  print("Residual Neural Network created.")
  return Model(inputs=model_input, outputs = x, name="MyResNet")
