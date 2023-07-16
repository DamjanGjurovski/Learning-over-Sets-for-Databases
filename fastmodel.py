import numpy as np
import tensorflow as tf
from keras import backend as K
import scipy
import time
import sys


def get_size(elem):
    sum_bytes = 0
    # if isinstance(elem, np.ndarray):
    #     sum_bytes = elem.nbytes
    if isinstance(elem, list):
        for e in elem:
            sum_bytes += sys.getsizeof(e)
    else:
        sum_bytes=sys.getsizeof(elem)
    return sum_bytes

class FastModel:
    def __init__(self, model, encode_layers_len, middle_layers_len, decode_layers_len, compression = False, ns = None, embed = True, compressed_dims = None, concat = True):
        self.embedding = []
        self.enc_w_len = []
        self.mid_w_len = []
        self.dec_w_len = []

        self.enc_weights = []
        self.enc_biases = []

        self.mid_weights = []
        self.mid_biases = []

        self.dec_weights = []
        self.enc_biases = []

        self.compression = compression
        self.ns = ns
        self.embed = embed
        self.compressed_dims = compressed_dims
        self.concat = concat
        if not compression:
            self.encode_layers_len = encode_layers_len
            self.middle_layers_len = 0
            self.decode_layers_len = decode_layers_len
        else:
            self.encode_layers_len = 0
            self.middle_layers_len = encode_layers_len
            self.decode_layers_len = decode_layers_len
        self.compression = compression
        self.create_model(model)

    def create_model(self, model):
        print("Creating a numpy version from the model")
        print_model = True
        if print_model:
            print("============")
            for layer in model.layers:
                print(layer.name)
                print(layer)
                print(layer.get_weights())
            print("============")
            print(model.summary())
            print("============")
        if not self.compression:
            print("Compressing")
            self.embedding = np.array(model.layers[1].weights[0])
            print(str("The embedding is ") + str(self.embedding))
            start = 2
        else:
            print("Not compressing")
            # we are skipping the ns layers and we are taking the embeddings
            start = self.ns
            end = start + self.ns
            # this is for embeddings
            for i in range(start, end):
                if self.embed:
                    print(i)
                    print(model.layers[i].name)
                    self.embedding.append(np.array(model.layers[i].weights[0]))

            # middle part is missing but we will add it
            start = end + 1

        end = start + self.encode_layers_len
        print("Start ")
        self.enc_weights, self.enc_biases = self.get_layers_weights(model, start, end)

        start = end
        if self.encode_layers_len > 0:
            start = start + 1
        end = start + self.middle_layers_len
        print("Middle")
        self.mid_weights, self.mid_biases = self.get_layers_weights(model, start, end)

        if self.encode_layers_len == 0 and self.middle_layers_len == 0:
            end = end + 1

        start = end
        if self.middle_layers_len > 0:
            start = start + 1
        end = start + self.decode_layers_len + 1
        print("End")
        self.dec_weights, self.dec_biases = self.get_layers_weights(model, start, end)

    def get_layers_weights(self, model, start, end):
        print(str(start) + "-" + str(end))
        if start == end:
            return None, None
        weights = []
        biases = []
        for i in range(start, end):
            print(i)
            print(model.layers[i].name)
            weights.append(np.array(model.layers[i].get_weights()[0]))
            biases.append(np.array(model.layers[i].get_weights()[1]))

        print(weights)
        print(biases)
        return weights, biases

    def predict(self, set):
        lambda_sum = None
        i = 0

        # Variant 1
        # for item in set:
        #     if lambda_sum is None:
        #         lambda_sum = self.embed_encode_layer(item)
        #     else:
        #         lambda_sum += self.embed_encode_layer(item)
        #     # print(lambda_sum)
        #     i += 1
        # Variant 1

        # Variant 2
        res = self.embed_encode_layer_multiple([int(i) for i in set])
        lambda_sum = np.sum(res, axis=0)
        # Variant 2
        for i in range(self.decode_layers_len):
            lambda_sum = np.matmul(lambda_sum, self.dec_weights[i]) + self.dec_biases[i]
            # relu
            lambda_sum *= (lambda_sum > 0)

        lambda_sum = np.matmul(lambda_sum, self.dec_weights[self.decode_layers_len]) + self.dec_biases[self.decode_layers_len]
        lambda_sum = 1 / (1 + np.exp(-lambda_sum))
        # relu
        # lambda_sum *= (lambda_sum > 0)
        # print(lambda_sum)
        return lambda_sum


    def embed_encode_layer(self, item):
        item = int(item)
        enc_w_out = self.embedding[item]
        for i in range(self.encode_layers_len):
            enc_w_out = np.matmul(enc_w_out, self.enc_weights[i]) + self.enc_biases[i]
            # relu
            enc_w_out *= (enc_w_out > 0)
        return enc_w_out

    def embed_encode_layer_multiple(self, items):
        enc_w_out = self.embedding[items]
        for i in range(self.encode_layers_len):
            enc_w_out = np.matmul(enc_w_out, self.enc_weights[i]) + self.enc_biases[i]
            # relu
            enc_w_out *= (enc_w_out > 0)
        return enc_w_out




    def embed_encode_layer_compression(self, item, i):
        # item = int(item) TODO IF NEEDED
        if self.embed:
            enc_w_out = self.embedding[i][item]
        else:
            enc_w_out = np.zeros(self.compressed_dims[i])
            if item != -1:
                enc_w_out[item] = 1

        for i in range(self.encode_layers_len):
            enc_w_out = np.matmul(enc_w_out, self.enc_weights[i]) + self.enc_biases[i]
            # relu
            enc_w_out *= (enc_w_out > 0)

        return enc_w_out

    def embed_encode_layer_compression_multiple(self, item, i):
        # item = int(item) TODO IF NEEDED
        if self.embed:
            enc_w_out = self.embedding[i][item]
        else:
            enc_w_out = np.zeros(self.compressed_dims[i])
            if item != -1:
                enc_w_out[item] = 1

        for i in range(self.encode_layers_len):
            enc_w_out = np.matmul(enc_w_out, self.enc_weights[i]) + self.enc_biases[i]
            # relu
            enc_w_out *= (enc_w_out > 0)

        return enc_w_out


    def embed_encode_layer_compression_multiple(self, items, i):
        # item = int(item) TODO IF NEEDED
        if self.embed:
            enc_w_out = self.embedding[i][items]
        else:
            # print(self.compressed_dims)
            # here we need to change it also
            len_items = len(items)
            enc_w_out = np.zeros([len_items, self.compressed_dims[i]])
            # print("Shape")
            # print(np.shape(enc_w_out))
            for i, item in enumerate(items):
                if item != -1:
                    enc_w_out[i][item] = 1


        for i in range(self.encode_layers_len):
            enc_w_out = np.matmul(enc_w_out, self.enc_weights[i]) + self.enc_biases[i]
            # relu
            enc_w_out *= (enc_w_out > 0)

        return enc_w_out






    def predict_compression(self, set):
        lambda_sum = None
        i = 0
        results = []

        # time_start = time.perf_counter_ns()
        set_ns_i = set[0]
        # print(set_ns_i)
        # # VARIANT 1 Much faster
        results = self.embed_encode_layer_compression_multiple(set_ns_i, 0)

        for ns_i in range(1, self.ns):
            set_ns_i = set[ns_i]
            partial = self.embed_encode_layer_compression_multiple(set_ns_i, ns_i)
            if self.concat:
                results = np.concatenate([results, partial], axis=1)
            else:
                results = np.add(results, partial)
        # #  VARIANT 1

        # VARIANT 2
        # for i, item in enumerate(set_ns_i):
        #     subpart_i = self.embed_encode_layer_compression(item, 0)  # np.array(self.embed_encode_layer_compression(item, ns_i)) # TODO REMOVED
        #     results.append(subpart_i)
        # for ns_i in range(1, self.ns):
        #     set_ns_i = set[ns_i]
        #     for i, item in enumerate(set_ns_i):
        #         subpart_i = self.embed_encode_layer_compression(item, ns_i)  # np.array(self.embed_encode_layer_compression(item, ns_i)) # TODO REMOVED
        #         if self.concat:
        #             results[i] = np.concatenate([results[i], subpart_i], axis=0)
        #         else:
        #             results[i] = np.add(results[i], subpart_i)
        # VARIANT 2

        # time_end = time.perf_counter_ns()
        # print("Encoding")
        # print((time_end - time_start) * 1e-6)


        # print(self.middle_layers_len)

        # VARIANT 1
        # print(np.shape(results))

        r = results
        for i in range(self.middle_layers_len):
            # print(np.shape(self.mid_weights[i]))
            r = np.matmul(r, self.mid_weights[i]) + self.mid_biases[i]
            # relu
            r *= (r > 0)
                # new_results.append(r)
        # print(new_results)
        # results = r
        lambda_sum = np.sum(r, axis=0)
        # VARIANT 1



        # VARIANT 2
        # new_results = []
        # for r in results:
        #     for i in range(self.middle_layers_len):
        #         r = np.matmul(r, self.mid_weights[i]) + self.mid_biases[i]
        #         # relu
        #         r *= (r > 0)
        #         new_results.append(r)
        # # print(new_results)
        # results = new_results
        # # here we need the sum and finally
        # lambda_sum = None
        # for item in results:
        #     if lambda_sum is None:
        #         lambda_sum = item
        #     else:
        #         lambda_sum += item
        # VARIANT 2


        # this part is the same
        for i in range(self.decode_layers_len):
            lambda_sum = np.matmul(lambda_sum, self.dec_weights[i]) + self.dec_biases[i]
            # relu
            lambda_sum *= (lambda_sum > 0)

        lambda_sum = np.matmul(lambda_sum, self.dec_weights[self.decode_layers_len]) + self.dec_biases[self.decode_layers_len]
        lambda_sum = 1 / (1 + np.exp(-lambda_sum))
        # relu
        # lambda_sum *= (lambda_sum > 0)
        # print(lambda_sum)
        return lambda_sum




    def get_size(self):
        total_size = get_size(self.embedding) + get_size(self.enc_w_len) + get_size(self.enc_w_len)\
                     + get_size(self.enc_w_len)\
                     + get_size(self.mid_w_len)\
                     + get_size(self.dec_w_len)\
                     + get_size(self.enc_weights)\
                     + get_size(self.enc_biases)\
                     + get_size(self.mid_weights)\
                     + get_size(self.mid_biases)\
                     + get_size(self.dec_weights)\
                     + get_size(self.enc_biases)\
                     + get_size(self.compression)\
                     + get_size(self.ns)\
                     + get_size(self.embed)\
                     + get_size(self.compressed_dims)\
                     + get_size(self.concat)
        return total_size
