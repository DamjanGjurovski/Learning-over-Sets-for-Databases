from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from deepset_model import deepset_model
import numpy as np
import time, os
from estimates_file import check_network_estimates
import store_statistics
from sklearn.utils import class_weight
from deepset_model_modified import get_deepset_model_modified
import pickle, sys, math
import error_calculator

# from new_deepset_model import get_deepset_model
def inverse_transform_minmax(y):
    """ Returns the inverse transform of the number """
    y = denormalize_MINMAX(y)
    y = K.exp(y)
    return y

def inverse_transform_minmax_single(y):
    """ Returns the inverse transform of the number """
    y = denormalize_MINMAX(y)
    y = math.exp(y)
    return y

def q_loss(y_true, y_pred):
    """ Calculation of q_loss with the original values """
    y_true = inverse_transform_minmax(y_true)
    y_pred = inverse_transform_minmax(y_pred)
    return K.maximum(y_true, y_pred) / K.minimum(y_true, y_pred)

def q_loss_single(y_true, y_pred):
    """ Calculation of q_loss with the original values """
    y_true = inverse_transform_minmax_single(y_true)
    y_pred = inverse_transform_minmax_single(y_pred)
    return max(y_true, y_pred) / min(y_true, y_pred)

def write_MIN_MAX(file_name, MIN, MAX):
    with open("minmax/" + file_name, "w") as f:
        f.write(str(int(MIN)) + " " + str(int(MAX)))

def normalize(y, min, max):
    """ Normalization used for the cardinality """
    y = (y - min) / (max - min)
    return y

def denormalize(y, min, max):
    """ Denormalization used for the cardinality """
    y = y * (max - min) + min
    return y

def transform_sets(sets, num_sets, max_train_length):
    X = np.zeros((num_sets, max_train_length))
    ix = 0
    for i in range(len(sets)):
        set_i = list(sets[i])
        n = len(set_i)
        for j in range(1, n + 1):
            X[ix, -j] = set_i[j - 1]
        ix += 1
    return X

def normalize_MINMAX(y):
    """ Normalization with MIN and MAX value """
    if 'MIN' not in globals():
        global MIN, MAX
        MIN = min(y)
        MAX = max(y)
    y = (y - MIN) / (MAX - MIN)
    return y

def denormalize_MINMAX(y):
    """ Denormalization with MIN and MAX value """
    y = (y * (MAX - MIN)) + MIN
    return y

def read_MIN_MAX(file_name):
    """ Reading of MIN and MAX value needed for prediction time since we require the same scaling as during training"""
    with open("minmax/" + file_name, "r") as f:
        line = f.readline()
        min_y = int(line.split(" ")[0])
        max_y = int(line.split(" ")[1])
        return min_y, max_y

def scale_cardinality(y, scale, do_training, dataset_name, cardinality = True, min_boundary = True):
    scaler_y = None
    print("Begin scaling")
    """ Other explored strategies, LMKG currently uses scale 3 """
    if scale == 0:
        print("We are using log1p")
        y = np.log(y + 1) + 1
    elif scale == 1:
        print("We are normalizing")
        global min_cardinality, max_cardinality
        min_cardinality = min(y)
        max_cardinality = max(y)
        y = normalize(y, min_cardinality, max_cardinality)
    elif scale == 2:
        scaler_y = MinMaxScaler()
        print("We are standardizing")
        y = np.log(y)
        scaler_y.fit(np.reshape(y, (-1, 1)))
        y = scaler_y.transform(np.reshape(y, (-1, 1)))
    elif scale == 3:
        if not do_training:
            global MIN, MAX
            """ We need the same min and max from the training data during evaluation """
            path_folder = ""
            if not cardinality:
                path_folder = "boundaries/"
                if min_boundary:
                    path_folder += "min_boundary"
                else:
                    path_folder += "max_boundary"

            MIN, MAX = read_MIN_MAX(path_folder + "min_max_" + dataset_name + ".txt")
            print("The min is " + str(MIN))
            print("The max is " + str(MAX))
            MIN = np.log(MIN)
            MAX = np.log(MAX)

        y = np.log(y)
        y = normalize_MINMAX(y)
    else:
        print("Scale not supported")
    return y, scaler_y

def unscale_cardinality_single(y, min_cardinality, max_cardinality, scale = 3, scaler_y = None):
    if scale == 0:
        y = np.round(np.exp(y-1)-1).astype(int)
    elif scale == 1:
        y = denormalize(y, min_cardinality, max_cardinality)
    elif scale == 2:
        y = np.reshape(y, (-1, 1))
        y = scaler_y.inverse_transform(y)
        y = np.exp(y)
    elif scale == 3:
        y = denormalize_MINMAX(y)
        y = np.exp(y)
        y = np.round(y)
    else:
        print("Scale not supported")
    return y

def unscale_cardinality(y, preds, scale, scaler_y):
    if scale == 0:
        preds = np.round(np.exp(preds-1)-1).astype(int)
        y = np.round(np.exp(y-1)-1).astype(int)
    elif scale == 1:
        preds = denormalize(preds, min_cardinality, max_cardinality)
        y = denormalize(y, min_cardinality, max_cardinality)
    elif scale == 2:
        preds = np.reshape(preds, (-1, 1))
        y = np.reshape(y, (-1, 1))
        preds = scaler_y.inverse_transform(preds)
        y = scaler_y.inverse_transform(y)
        preds = np.exp(preds)
        y = np.exp(y)
    elif scale == 3:
        preds = denormalize_MINMAX(preds)
        y = denormalize_MINMAX(y)
        preds = np.exp(preds)
        y = np.exp(y)
        preds = np.round(preds)
        y = np.round(y)
    else:
        print("Scale not supported")
    return y, preds


def create_model(max_length, max_elem, encode_layers = [], middle_layers=[], decode_layers = [],  deepset = True, embed_size=32, compression = False, compressed_dims = None, embed = True, ns = 2, encode_dimensions=0, one_hot = True, concat = True):
    if deepset:
        if not compression:
            model = deepset_model(max_length, max_elem, embed_size=embed_size, encode_layers=encode_layers, decode_layers=decode_layers)
        else:
            model = get_deepset_model_modified(max_length, compressed_columns_dim = compressed_dims, embed_size=embed_size, encode_layers=[], middle_layers= encode_layers, decode_layers=decode_layers, compile=False, ns = ns, one_hot = one_hot, concat = concat)
        print(model.summary())
    else:
        print("Implement any other model")
        exit(1)
    return model

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def train_eval_model(model, X, y, dataset_name, batch_size, epochs, scale = 3, do_training = False, encoding_avg_time = 0, statistics_file_name = "stats", classification = False, cardinality = True, min_boundary = True, path = "", boundaries = False, ns = 2, compression = False, error=0, sets = None,
                                   outlierremove = False,
                                   start = None, step = None, boundary = None):
    original_y = y.copy()
    real_y = y.copy()
    if do_training and not classification:
        path_folder = ""
        if not cardinality:
            path_folder = "boundaries/"
            if min_boundary:
                path_folder += "min_boundary"
            else:
                path_folder += "max_boundary"
        min_y = min(y)
        max_y = max(y)
        write_MIN_MAX(path_folder + "min_max_" + dataset_name + ".txt", min_y, max_y)

    if not classification:
        y, scaler_y = scale_cardinality(y, scale, do_training, dataset_name, cardinality = cardinality)

    start_time = time.time()
    if do_training:
        if classification:
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y),
                                                              y=y)  # compute_class_weight('balanced', np.unique(y), y)
            class_weights = {i: class_weights[i] for i in range(2)}
            model.fit(x=X, y=y, #validation_split=0.2,
                      class_weight=class_weights,
                      epochs=epochs, batch_size=batch_size, shuffle = True
                      # ,callbacks=[CustomCallback(model, X, y)]
                      )
        else:
            if not outlierremove:
                model.fit(x=X, y=y, # validation_split=0.2,
                          epochs=epochs, batch_size=batch_size, shuffle = True
                          )
            else:
                from custom_sequence import DataGen
                train_ds = DataGen(X, y, batch_size, model, ns = ns, compression = compression, error = error, sets = sets,
                                   outlierremove = outlierremove,
                                   start = start, step = step, boundary = boundary, real_y = real_y, min_y = min_y, max_y = max_y, dataset_name = dataset_name)

                model.fit(train_ds,
                          epochs=epochs, batch_size=batch_size,
                          steps_per_epoch = None
                          )

                y_outliers = train_ds.set_preds

                len_outliers = len(train_ds.set_outliers)
                outliers_map = dict()
                for i in range(len_outliers):
                    outliers_map[train_ds.set_outliers[i]] = int(y_outliers[i])

                print(sys.getsizeof(outliers_map))
                with open("edbt_boundary_maps/" + path + statistics_file_name + '_dict.pickle', 'wb') as handle:
                    pickle.dump(outliers_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Saving at location")
                print("edbt_boundary_maps/" + path + statistics_file_name + '_dict.pickle')
                X = [train_ds.x[i] for i in range(train_ds.x.shape[0])]
                y = train_ds.y

    if (not classification) and not do_training:
        fname = "edbt_boundary_maps/"  + path + statistics_file_name + '_dict.pickle'
        if os.path.isfile(fname):
            with open(fname, 'rb') as handle:
                outliers_map = pickle.load(handle)
            print(sys.getsizeof(outliers_map))
        else:
            outliers_map = dict()
        return outliers_map, MIN, MAX

    end_time = time.time() - start_time
    print('The time needed for training %.3f seconds \n' % end_time)
    avg_time_prediction_ms = 0
    if do_training:
        if not classification:
            if len(np.shape(X)) == 3 or not outlierremove:
                preds = model.predict(X)
            else:
                print(np.shape(train_ds.x))
                preds = model.predict(train_ds.x)
        else:
            preds = model.predict(X)
    else:
        total_pred = 0
        repetitions = 1
        for _ in range(repetitions):
            predictions_start_time = time.time()
            preds = model.predict(X)
            predictions_start_time = time.time() - predictions_start_time
            total_pred += predictions_start_time / len(y)
        if repetitions != 0:
            avg_time_prediction_ms = (total_pred / repetitions) * 1000.0
            print("The average prediction time is " + str(avg_time_prediction_ms))

    if not classification:
        y, preds = unscale_cardinality(y, preds, scale, scaler_y)

    # removing outliers if needed
    nb_outliers = 0
    if nb_outliers > 0:
        sorted_y = y
        sorted_y = sorted(sorted_y, reverse = True)
        outliers = sorted_y[:nb_outliers]
        preds_outlier = []
        y_outlier = []
        for i in range(len(y)):
            if y[i] not in outliers:
                y_outlier.append(y[i])
                preds_outlier.append(preds[i])
        y = y_outlier
        preds = preds_outlier

    ''' Accuracy statistics for regression '''
    if not classification:
        correct_estimate = store_statistics.check_network_estimates(preds, y, statistics_file_name = statistics_file_name, print_time=0, nb_outliers = nb_outliers, avg_time_prediction_ms = avg_time_prediction_ms, path = path)
        error_calculator.store_error(y, preds, length_of_range=100, name=statistics_file_name, dataset_name = dataset_name, training=do_training, path = path)


    if (not classification) and outlierremove:
        with open("edbt_boundary_maps/" + path + statistics_file_name + '_dict.pickle', 'rb') as handle:
            outliers_map = pickle.load(handle)
        print(sys.getsizeof(outliers_map))
        print("edbt_boundary_maps/" + path + statistics_file_name + '_dict.pickle')
        return outliers_map, MIN, MAX


def store_model(nn_model, save_path, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    nn_model.save(save_path + model_name)
    '''storing just the weights of the model'''
    weights = nn_model.get_weights()
    if not os.path.exists(save_path + '/weights/'):
        os.makedirs(save_path + '/weights/')
    with open(save_path + '/weights/' + model_name + '_size.txt', "w") as f:
        f.write(str(weights))
