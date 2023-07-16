import store_statistics
from data_processor_set import read_transform_sets_cardinality, read_sets
from helper_methods import *
from deepset_model import *
import params, time, random
import numpy as np
from keras.models import load_model
from setindex import SetIndex
import tensorflow as tf

with tf.device("CPU:0"):
    if __name__ == '__main__':
        partial_path = ""

        min_boundary = True
        args, model_name, encode_layers, middle_layers, decode_layers, sv_d, ns, boundaries = params.params()
        create_new_model = args.training
        avg_time_encoding_ms = time.time()
        if boundaries:
            boundary_type = "min_boundary" if min_boundary else "max_boundary"
            model_name += "_" + str(boundary_type) + ""
            path = "data/" + args.dataset_name + "/" + boundary_type + ".txt"
        else:
            path = "data/" + args.dataset_name + "/cardinality_max_size5.txt"
            full = True
            if full:
                model_name += "_full"
            else:
                path = "data/" + args.dataset_name + "/cardinality_apriori_threshold_2.txt"
                model_name += "_arm"
        original_dataset_path = "data/" + args.dataset_name + "/" + args.dataset_name + "_int.csv"
        model_name += '.h5'
        if args.training:
            X, y, x_dim_max, sets, set_card_map = read_transform_sets_cardinality(path, max_train_length=args.max_length,
                                                              compression=args.compression, compression_sv_d=sv_d, ns=ns,
                                                              one_hot=args.onehot)
        else:
            # data for evaluation, currently set on the same data you can change accordingly
            X, y, x_dim_max, sets, set_card_map = read_transform_sets_cardinality(path, max_train_length=args.max_length,
                                                              compression=args.compression, compression_sv_d=sv_d, ns=ns,
                                                              one_hot=args.onehot)
            original_sets, _ = read_sets(original_dataset_path)

        avg_time_encoding_ms = (time.time() - avg_time_encoding_ms)/ len(y)

        '''create the model'''
        if boundaries:
            save_path = 'edbt_models/' + args.dataset_name + "/" + "boundaries/"
            partial_path = args.dataset_name + "/" + "boundaries/"
        else:
            save_path = 'edbt_models/' + args.dataset_name + "/" + "card/"
            partial_path = args.dataset_name + "/" + "card/"

        compressed_dims = [(i+1) for i in x_dim_max]
        total_encode_dims = sum(compressed_dims) + ns + 1
        print("The ns is " + str(ns))
        print("The compressed dims are " + str(compressed_dims))
        print("The total encode dims is " + str(total_encode_dims))
        print("The shape of the data is")
        print(np.shape(X))
        if create_new_model:
            nn_model = create_model(max_length=args.max_length, encode_layers=encode_layers, middle_layers=middle_layers,
                                    decode_layers=decode_layers, max_elem=args.max_elem, embed_size=args.embed_size,
                                    compression=args.compression, compressed_dims=compressed_dims, ns=ns,
                                    encode_dimensions=total_encode_dims, one_hot=args.onehot, concat = args.concat)
        else:
            if args.error == 0:
                nn_model = load_model(save_path + model_name, custom_objects={"q_loss": q_loss})
            else:
                nn_model = load_model(save_path + model_name)

        if args.training:
            compile_model(nn_model, classification=False, decay=False, learning_rate=0.0002, error=args.error)

        if not args.training:
            '''train the model'''
            outliers_map, min_cardinality, max_cardinality = train_eval_model(nn_model, X, y, args.dataset_name, args.batch_size, args.epochs, scale=args.scale,
                                        do_training=args.training, encoding_avg_time=avg_time_encoding_ms,
                                        statistics_file_name='stats-' + model_name + ".txt", cardinality = not boundaries, min_boundary=min_boundary, path = partial_path, boundaries = boundaries, ns = ns, compression=args.compression, error=args.error, sets = sets,
                                   outlierremove = args.outlierremoval,
                                   start = args.startremoval, step = args.stepremoval, boundary = args.boundaryremoval)
        else:
            '''test the model'''
            train_eval_model(nn_model, X, y, args.dataset_name, args.batch_size, args.epochs, scale=args.scale,
                             do_training=args.training, encoding_avg_time=avg_time_encoding_ms,
                             statistics_file_name='stats-' + model_name + ".txt", cardinality=not boundaries, min_boundary=min_boundary,
                             path=partial_path, boundaries=boundaries, ns = ns, compression=args.compression, error=args.error, sets = sets,
                             outlierremove=args.outlierremoval,
                             start=args.startremoval, step=args.stepremoval, boundary=args.boundaryremoval
                             )

        '''storing of the model'''
        if args.training:
            store_model(nn_model, save_path, model_name)

        total_time_model, total_time_fast_model, total_time_fast_model_complete, total_time_subset = 0, 0, 0, 0
        count_from_model = 0
        if not args.training:

            si = SetIndex(nn_model, outliers_map, min_cardinality, max_cardinality, len(encode_layers), len(middle_layers), len(decode_layers), max_length = args.max_length, compression=args.compression, ns = ns, one_hot = args.onehot, sv_d = sv_d, compressed_dims = compressed_dims, errors_name=model_name, concat=args.concat, original_sets = original_sets, outlierremoval = args.outlierremoval, path = partial_path)
            si_size = si.get_size()
            len_X = len(sets)
            print("Len of sets: " + str(len_X))
            problems = 0
            sets1 = sets.copy()
            random.Random(8).shuffle(sets1)
            len_X = 10000
            if args.compression:
                X = np.array(X)

            if boundaries:
                for i in range(0, len_X):
                    if i % 10000 == 0:
                        print(str(i) + "/" + str(len_X))
                    set_i = np.array(list(sets1[i]))
                    time_start = time.perf_counter_ns()
                    idx, true_res = si.equality_subset_predict(set_i, equality=False)
                    time_end = time.perf_counter_ns()
                    if idx == -1:
                        problems += 1
                        print("Item not present")
                    if not true_res:
                        total_time_subset += (time_end - time_start)
                        count_from_model += 1

                print("Not found: " + str(problems))
            else:
                y_true = []
                y_pred = []
                count_from_model = 0
                for i in range(0, len_X):
                    set_i = np.array(list(sets1[i]))
                    time_start = time.perf_counter_ns()
                    idx, res = si.full_predict(set_i, unscale=True, full_predict=True)
                    time_end = time.perf_counter_ns()
                    total_time_fast_model_complete += (time_end - time_start)
                    if not res:
                        count_from_model += 1
                    y_true.append(set_card_map[sets1[i]])
                    y_pred.append([idx])

                avg_time_prediction_ms = (total_time_fast_model_complete / len_X) * 1e-6
                print("The average prediction time is")
                print(avg_time_prediction_ms)
                # Uncomment for further statistics of the hybrid structure
                # statistics_file_name = "edbt_statistics_hybrid/" + partial_path + model_name + ".txt"
                # correct_estimate = store_statistics.check_network_estimates(y_pred, y_true,
                #                                                             statistics_file_name=statistics_file_name,
                #                                                             print_time=0, nb_outliers=0,
                #                                                             avg_time_prediction_ms=avg_time_prediction_ms)
                # print(len(y_true))


            if si.time_outliers != 0:
                time_outliers = (si.time_outliers/count_from_model) * 1e-6
            else:
                time_outliers = 0
            time_model = (si.time_model/count_from_model) * 1e-6
            time_errors = (si.time_errors/count_from_model) * 1e-6
            print("Time learned model: " + str(time_model))
            print("Time outliers: " + str(time_outliers))
            print("Time error: " + str(time_errors))
            print("Resulting in total time of")
            print(time_outliers + time_model + time_errors)
            print((total_time_subset / count_from_model) * 1e-6)
            print("Queries from model " + str(count_from_model) + ", total " + str(len_X))

