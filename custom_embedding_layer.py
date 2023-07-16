import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Lambda
from keras import backend as K

def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    def _one_hot(x, num_classes):
        # before the model was with uint8 and the -1 was some random value since it was given a unsigned value
        # K.cast(x, 'uint8')
        # print( K.print_tensor(x, message='y_true = '))
        return K.one_hot(K.cast(x, 'int32'),
                          num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))


class MultihotEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        super(MultihotEmbedding, self).__init__(**kwargs)

    def call(self, x):
        self.get_embeddings = K.one_hot(x, num_classes=self.vocab_size)
        self.reduce_embeddings = K.sum(self.get_embeddings,axis = -2)
        return self.reduce_embeddings

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.vocab_size)

