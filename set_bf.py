from data_processor_set import read_transform_sets_bf, transform_sets_individual
from helper_methods import *
from deepset_model import *
import params, time
from sklearn.utils import shuffle
# from CLBF import CLBF
from data_processor_set import transform_single_set, transform_single_set_batch
from fastmodel import FastModel
import numpy as np
from data_processor_set import read_sets
from setindex import SetIndex
from keras.models import load_model
import tensorflow as tf

with tf.device("CPU:0"):
    if __name__ == '__main__':
        args, model_name, encode_layers, middle_layers, decode_layers, sv_d, ns, _ = params.params()
        create_new_model = args.training
        avg_time_encoding_ms = time.time()
        path = "data/" + args.dataset_name + "/"
        negative_path = path + "bf_negative.txt"
        full = True
        if full:
            positive_path = path + "bf_positive.txt"
            model_name += "_full"
        else:
            positive_path = path + args.dataset_name + "_int.csv"
            model_name += "_partial"

        model_name += "_negative1000000"
        model_name += '.h5'
        print(model_name)
        if args.training:
            X, y, x_dim_max = read_transform_sets_bf(positive_path, negative_path, max_set_size=args.max_length, compression=args.compression, compression_sv_d=sv_d, ns = ns, one_hot = args.onehot)
        else:
            # define different testing paths if you want
            # positive_path = ""
            # negative_path = ""
            X, y, x_dim_max = read_transform_sets_bf(positive_path, negative_path, max_set_size=args.max_length, compression=args.compression, compression_sv_d=sv_d, ns = ns, one_hot = args.onehot)
        avg_time_encoding_ms = (time.time() - avg_time_encoding_ms)/ len(y)

        # we cannot do it the other way around
        if not args.compression:
            X, y = shuffle(X, y, random_state=0)

        '''create the model'''
        save_path = 'edbt_models/' + args.dataset_name + "/" + "bf/"
        print(args.embed_size)
        compressed_dims = [(i + 1) for i in x_dim_max]

        print("The compressed dims are: " + str(compressed_dims))
        total_encode_dims = sum(compressed_dims) + ns + 1
        if create_new_model:
            nn_model = create_model(max_length=args.max_length, encode_layers=encode_layers, middle_layers=middle_layers,
                         decode_layers=decode_layers, max_elem=args.max_elem, embed_size=args.embed_size,
                         compression=args.compression, compressed_dims=compressed_dims, ns=ns,
                         encode_dimensions=total_encode_dims, one_hot=args.onehot)
        else:
            nn_model = load_model(save_path + model_name)

        compile_model(nn_model, classification=True, decay=True, learning_rate=1e-5)

        '''train the model'''
        train_eval_model(nn_model, X, y, args.dataset_name, args.batch_size, args.epochs, scale = args.scale, do_training=args.training, encoding_avg_time=avg_time_encoding_ms, statistics_file_name = 'statistics-'+ model_name + ".txt", classification=True, path = args.dataset_name + "/" + "bf/")

        '''storing of the model'''
        if args.training:
            store_model(nn_model, save_path, model_name)

        print("The model is trained and saved, to test it set the parameter training to False, and use the same exact parametars as before.")

        '''testing of the model'''
        if not args.training:
            print(positive_path)
            original_sets, _ = read_sets(positive_path)
            len_X = 1000

            si = SetIndex(nn_model, None, 0, 1, len(encode_layers),
                          len(middle_layers), len(decode_layers), max_length=args.max_length,
                          compression=args.compression, ns=ns, one_hot=args.onehot, sv_d=sv_d,
                          compressed_dims=compressed_dims, errors_name=model_name, concat=args.concat,
                          original_sets=original_sets, outlierremoval=False, path="partial_path", classification = True)

            total_time = 0
            for i in range(0, len_X):
                if i % 10000 == 0:
                    print(str(i) + "/" + str(len_X))
                item = np.array(list(original_sets[i]))
                time_start = time.perf_counter_ns()
                idxMain, true_res = si.predict_bf(item)
                time_end = time.perf_counter_ns()
                total_time += (time_end - time_start)

            total_time = (total_time / len_X) * 1e-6
            print("Time learned model: " + str(total_time))

            # guidance for creating the complete bf
            # dp = CLBF(nn_model, 0.01, model_name="deepset")
            # dp.save_backup_bloom_filter(path)
            # dp.create_bloom_filter(X,y)
            # dp.create_complete(X, y)

