import keras.backend as K
from keras.layers import Input, Dense, LSTM, GRU, Embedding, Lambda
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras import metrics, losses
import helper_methods


METRICS_CLASSIFICATION = [
            metrics.TruePositives(name='tp'),
            metrics.FalsePositives(name='fp'),
            metrics.TrueNegatives(name='tn'),
            metrics.FalseNegatives(name='fn'),
            metrics.BinaryAccuracy(name='binary_accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(name='auc'),
            metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
            'accuracy',
            'binary_crossentropy'
        ]
METRICS_REGRESSION = ['mae', 'mse', 'accuracy']

def define_loss_metrics(classification, error = 0):
    if classification:
        loss = "binary_crossentropy"
        METRICS = METRICS_CLASSIFICATION
    else:
        if error == 0:
            print("Training with q loss.")
            loss = helper_methods.q_loss
        else:
            print("Training with mae.")
            loss = "mae"
        METRICS = METRICS_REGRESSION
    return loss, METRICS

def compile_model(model, classification = True, learning_rate = 1e-3, decay = True, error = 0):
    loss, METRICS = define_loss_metrics(classification, error=error)
    opt = Adam(lr=learning_rate, decay=learning_rate / 300) if decay else Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss = loss, metrics = METRICS)

def deepset_model(max_length, max_elem, embed_size = 32, encode_layers = [], decode_layers = [64, 64]):
    input_txt = Input(shape=(max_length,))
    x = Embedding(max_elem, embed_size, mask_zero=True)(input_txt)
    for e_l in encode_layers:
        x = Dense(e_l, activation='relu')(x)
    Adder = Lambda(lambda x: K.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    for d_l in decode_layers:
        x = Dense(d_l, activation="relu")(x)
    encoded = Dense(1, activation="sigmoid")(x)
    summer = Model(input_txt, encoded)
    return summer

def gru_model(max_length, max_elem, embed_size = 32, encode_layers = [], decode_layers = [64, 64], classification = True):
    input_txt = Input(shape=(max_length,))
    x = Embedding(max_elem, embed_size, mask_zero=True)(input_txt)
    for e_l in encode_layers:
        x = Dense(e_l, activation='tanh')(x)
    x = GRU(80)(x)
    for d_l in decode_layers:
        x = Dense(d_l, activation="tanh")(x)
    encoded = Dense(1, activation="sigmoid")(x)
    summer = Model(input_txt, encoded)
    return summer

def lstm_model(max_length, max_elem, embed_size = 32, encode_layers = [], decode_layers = [64, 64], classification = True):
    input_txt = Input(shape=(max_length,))
    x = Embedding(max_elem, embed_size, mask_zero=True)(input_txt)
    for e_l in encode_layers:
        x = Dense(e_l, activation='tanh')(x)
    x = LSTM(50)(x)
    for d_l in decode_layers:
        x = Dense(d_l, activation="tanh")(x)
    encoded = Dense(1, activation="sigmoid")(x)
    summer = Model(input_txt, encoded)
    return summer