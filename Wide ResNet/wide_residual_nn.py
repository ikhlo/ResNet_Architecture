from keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Dropout, Add, AveragePooling2D


CHANNEL_AXIS = 1 if K.image_data_format() == "channels_first" else -1

def wide_res_identity_block(x, base, k, dropout_rate=0.0):

  x_init = x

  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)
  x = Conv2D(base * k, kernel_size=3, padding="same", kernel_initializer='he_normal', use_bias=False)(x)

  x = Dropout(dropout_rate)(x)

  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)
  x = Conv2D(base * k, kernel_size=3, padding="same", kernel_initializer='he_normal', use_bias=False)(x)

  x = Add()([x, x_init])
  return x


def wide_res_conv_block(x, base, k, s):

  x_init = x

  x = Conv2D(base * k, kernel_size=3, padding="same", strides=s, kernel_initializer='he_normal', use_bias=False)(x)
  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)
  
  x = Conv2D(base * k, kernel_size=3, padding="same", kernel_initializer='he_normal', use_bias=False)(x)

  x_init = Conv2D(base * k, kernel_size=3, padding="same", strides=s, kernel_initializer='he_normal', use_bias=False)(x_init)

  x = Add()([x, x_init])
  return x


def wide_resnet_model(input:np.ndarray, nb_of_classes: int, depth:int=16, k:int=8, dropout_rate:float=0.0):

  N = int((depth - 4) / 6)
  filters = (16, 32, 64)

  model_input = Input(shape=input.shape)
  x = Conv2D(16, kernel_size=3, padding="same", kernel_initializer='he_normal', use_bias=False)(model_input)
  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)
  
  for i in range(len(filters)):
    for j in range(N):
      if i == 0 and j == 0 : x = wide_res_conv_block(x, filters[i], k, 1)
      elif j == 0 : x = wide_res_conv_block(x, filters[i], k, 2)
      else : x = wide_res_identity_block(x, filters[i], k, dropout_rate)
    
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation('relu')(x)
  
  x = AveragePooling2D((8, 8), strides=(1, 1), padding="same")(x)
  x = Flatten()(x)

  x = Dense(nb_of_classes, activation='softmax')(x)

  print(f"Wide Residual Neural Network-{depth}-{k} created.")
  return Model(inputs=model_input, outputs = x, name=f"WideResNet-{depth}-{k}")