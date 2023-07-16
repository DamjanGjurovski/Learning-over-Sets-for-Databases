import os.path
import pickle
import math

def store_error(y, y_est, length_of_range, name, min_val = 0, dataset_name = "", training = False, path = ""):
    name = name.replace("stats-", "").replace(".h5","").replace(".txt", "")
    fname = name + "_error_" + str(length_of_range) + ".pickle"
    path = "edbt_error_maps/" + path + "/"
    path = path.replace("//","/")
    fname = path + fname
    print(fname)
    if os.path.isfile(fname) and not training:
        print("Not creating a dictionary. It already exists.")
        # load
        with open(fname, "rb") as handle:
            partition_error = pickle.load(handle)
    else:
        print("Creating a dictionary.")
        partition_error = dict()
        len_y = len(y)
        for i in range(len_y):
            y_i = y[i]
            y_est_i = int(y_est[i][0])

            partition_id = partition(y_est_i, min_val=min_val, length_of_range=length_of_range)
            error_i = int(math.ceil(abs(y_i - y_est_i)))
            print(str(y_i) + " " + str(y_est_i) + " "+ str(error_i))
            if partition_id in partition_error:
                partition_error[partition_id] = max(partition_error[partition_id], error_i)
            else:
                partition_error[partition_id] = error_i
        # save pickle
        print(partition_error)
        with open(fname, 'wb') as handle:
            pickle.dump(partition_error, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return partition_error

def partition(number, min_val, length_of_range):
    partition = (number - min_val) / length_of_range + 1
    return int(partition)
