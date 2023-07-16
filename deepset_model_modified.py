import numpy as np
import keras.backend as K
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import *
from custom_embedding_layer import OneHot

def gen_test_data(num_examples, length):
    Y = np.zeros((num_examples, length))
    sum_Y = np.zeros((num_examples))
    for i in range(num_examples):
        for j in range(1,length+1):
            Y[i,-j] = np.random.randint(1,10)
        sum_Y[i] = np.sum(Y[i])
    return Y, sum_Y

def get_deepset_model(max_length, max_elem):
    input_txt = Input(shape=(max_length,))
    x = Embedding(max_elem, 32, mask_zero=True)(input_txt)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    encoded = Dense(1)(x)
    summer = Model(input_txt, encoded)
    adam = Adam(lr=1e-4, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    return summer

def get_deepset_model_modified(max_length, embed_size = 4, compressed_columns_dim = [100, 100], encode_layers = [],  middle_layers= [30], decode_layers = [], compile = False, sigmoid = True, ns = 2, one_hot = True, concat = True):
    if ns == 2:
        nn1 = create_shared_embedding(max_length, compressed_columns_dim[0], embed_size = embed_size, encode_layers = encode_layers, one_hot = one_hot)
        nn2 = create_shared_embedding(max_length, compressed_columns_dim[1], embed_size = embed_size, encode_layers = encode_layers, one_hot = one_hot)
        if concat:
            combinedInput = concatenate([nn1.output, nn2.output])
        else:
            combinedInput = Add()([nn1.output, nn2.output])
    else:
        combinedInput_arr = []
        combinedModelInput_arr = []
        for i in range(ns):
            n_i = create_shared_embedding(max_length, compressed_columns_dim[i], embed_size=embed_size, encode_layers=encode_layers, one_hot = one_hot)
            combinedInput_arr.append(n_i.output)
            combinedModelInput_arr.append(n_i.input)
        if concat:
            combinedInput = concatenate(combinedInput_arr)
        else:
            combinedInput = Add()(combinedInput_arr)

    len_middle_layers = len(middle_layers)

    for i in range(0, len_middle_layers):
        combinedInput = Dense(middle_layers[i], activation="relu")(combinedInput)

    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(combinedInput)

    len_decode_layers = len(decode_layers)
    for i in range(0, len_decode_layers):
        x = Dense(decode_layers[i], activation="relu")(x)

    if sigmoid:
        encoded = Dense(1, activation="sigmoid")(x)
    else:
        encoded = Dense(1)(x)

    if ns == 2:
        model = Model(inputs=[nn1.input, nn2.input], outputs=encoded)
    else:
        model = Model(inputs = combinedModelInput_arr, outputs=encoded)

    if compile:
        adam = Adam(lr=1e-4, epsilon=1e-3)
        model.compile(optimizer=adam, loss='mae')
    return model

def create_shared_embedding(max_length, max_elem, embed_size = 32, encode_layers = [], one_hot = True):
     model = Sequential()
     model.add(Input(shape=(max_length,)))
     if one_hot:
        model.add(OneHot(input_dim=max_elem, input_length=max_length))
     else:
        model.add(Embedding(max_elem + 1, embed_size, mask_zero=True, trainable=True))
     len_additional_layers = len(encode_layers)
     for i in range(0, len_additional_layers):
         model.add(Dense(encode_layers[i], activation="relu"))
     return model