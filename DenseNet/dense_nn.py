from keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Add, AveragePooling2D, GlobalAveragePooling2D, Concatenate

CHANNEL_AXIS = 1 if K.image_data_format() == "channels_first" else -1

def dense_bottleneck_layer(x, nb_channels, dropout_rate):

  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)
  x = Conv2D(4 * nb_channels, kernel_size=(1,1), padding="same", kernel_initializer='he_normal', use_bias=False)(x)
  x = Dropout(dropout_rate)(x)

  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)
  x = Conv2D(nb_channels, kernel_size=(3,3), padding="same", kernel_initializer='he_normal', use_bias=False)(x)
  x = Dropout(dropout_rate)(x)
  
  return x


def dense_block(x, nb_channels, dropout_rate, nb_bottleneck_layer=5):

  all_layers = list()
  all_layers.append(x)

  for i in range(nb_bottleneck_layer):
    x = dense_bottleneck_layer(x, nb_channels, dropout_rate)
    all_layers.append(x)
    x = Concatenate(axis=CHANNEL_AXIS)(all_layers)
  
  return x


def dense_transition_layer(x, nb_channels, dropout_rate):

  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)
  x = Conv2D(nb_channels, kernel_size=(1,1), padding="same", kernel_initializer='he_normal', use_bias=False)(x)
  x = Dropout(dropout_rate)(x)
  x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)

  return x


def densenet_model(input_shape:tuple, nb_of_classes:int, nb_dense_block:int=4, dropout_rate:float=0.2, nb_filters_list:list=[6, 12, 24, 16]):


  model_input = Input(shape=input_shape)
  nb_channel = model_input.shape[CHANNEL_AXIS]


  x = Conv2D(2 * nb_channel, kernel_size=(3,3), padding="same", kernel_initializer='he_normal', use_bias=False)(model_input)
  x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

  for i in range(nb_dense_block-1):
    x = dense_block(x, nb_channel, dropout_rate, nb_filters_list[i])
    x = dense_transition_layer(x, nb_channel, dropout_rate)
  
  x = dense_block(x, nb_channel, dropout_rate, nb_filters_list[-1])

  x = BatchNormalization(axis=CHANNEL_AXIS)(x)
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  x = Dense(nb_of_classes)(x)

  print(f"Dense Neural Network-{sum(nb_filters_list)*2+5} created.")
  return Model(inputs=model_input, outputs = x, name=f"DenseNet-{sum(nb_filters_list)*2+5}")