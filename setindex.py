import math
import random
# from lib.utils import *
import mmh3
import numpy as np
from helper_methods import unscale_cardinality_single
# from helper_methods import inverse_transform_single
from data_processor_set import transform_single_set, transform_single_set_batch
import error_calculator
import time
import sys
from competitors_code import B_Tree_own

from fastmodel import FastModel

class SetIndex(object):
    def __init__(self, model, outliers_map, min_cardinality, max_cardinality, encode_layers_len, middle_layers_len, decode_layers_len, compression, max_length, ns = None, one_hot = False, sv_d = 0, compressed_dims = None, errors_name = "", concat = True, original_sets = None, outlierremoval = False, path = "", classification = False):
        self.original_sets = np.array(original_sets).tolist()
        self.model = model
        self.threshold = None
        self.outlier_removal = outlierremoval
        if self.outlier_removal:
            print("Len outlier")
            print(len(outliers_map.keys()))
            self.btree_flag = False
            if self.btree_flag:
                self.outliers_map = dict()
                self.create_BTree(outliers_map)
                self.outliers_map = outliers_map
            else:
                # SET 2 for cardinaltiy estimation
                self.hm_version = 2
                if self.hm_version == 1:
                    self.outliers_map = dict()
                    for key in outliers_map:
                        set_i = key
                        value = outliers_map[key]
                        # The other option is abs(hash(set_i)) % 10000000 allowing duplicate keys
                        set_hash = list(set_i)
                        np.sort(set_hash)
                        if set_hash in self.outliers_map:
                            list_hash = self.outliers_map[set_hash]
                            list_hash.append(value)
                            self.outliers_map[set_hash] = list_hash
                        else:
                            self.outliers_map[set_hash] = [value]
                else:
                    self.outliers_map = outliers_map

        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

        self.fastmodel = FastModel(model, encode_layers_len, middle_layers_len, decode_layers_len, compression, ns = ns, embed = not one_hot, compressed_dims = compressed_dims, concat = concat)
        if not classification:
            self.errors = error_calculator.store_error(None, None, length_of_range=100, name = errors_name, path = path)
        self.compression = compression
        self.ns = ns
        self.one_hot = one_hot
        self.sv_d = sv_d
        self.max_length = max_length
        self.time_outliers = 0
        self.time_model = 0
        self.time_errors = 0


    def get_size(self):
        print("Details for the model size")
        print(self.fastmodel.get_size() * 1e-6)
        print(sys.getsizeof(self.fastmodel) * 1e-6)
        print("Details for the errors")
        print(sys.getsizeof(self.errors) * 1e-6)
        if self.outlier_removal:
            print("Details for the outliers")
            print(len(self.outliers_map.keys()))
            print(sys.getsizeof(self.outliers_map) * 1e-6)
            if self.btree_flag:
                print("Details for the B+Tree")
                print(sys.getsizeof(self.b) * 1e-6)
                print("B- Tree size GSO")
                print(self.b.get_size(gso=True) * 1e-6)
                print("B- Tree size NO GSO")
                print(self.b.get_size(gso=False) * 1e-6)
            else:
                print("Details for the outlier map: " + str(sys.getsizeof(self.outliers_map) * 1e-6) + " MB")
                total_custom_size_of = 0
                total_bytes = 0
                for key in self.outliers_map.keys():
                    total_custom_size_of += sys.getsizeof(key)
                    total_custom_size_of += sys.getsizeof(self.outliers_map[key])
                    total_bytes += len(self.outliers_map[key]) if self.hm_version == 1 else 1
                total_bytes *= 4
                total_bytes += len(self.outliers_map.keys()) * 4 if self.hm_version == 1 else len(self.outliers_map.keys()) * 8
                print("CUSTOM FUNCTION SIZE " + str(total_bytes * 1e-6) + " MB")
                print("COUNTING " + str(total_custom_size_of * 1e-6))
                print("Nb keys: " + str(len(self.outliers_map.keys())))

        total_size = self.fastmodel.get_size() + sys.getsizeof(self.errors)\
                     + sys.getsizeof(self.threshold)\
                     + sys.getsizeof(self.min_cardinality)\
                     + sys.getsizeof(self.max_cardinality)\
                     + sys.getsizeof(self.compression)\
                     + sys.getsizeof(self.ns)\
                     + sys.getsizeof(self.one_hot)\
                     + sys.getsizeof(self.sv_d)\
                     + sys.getsizeof(self.max_length)
        # if self.outlier_removal:
        #     total_size += sys.getsizeof(self.outliers_map)
        return total_size

    def check_model(self, item):
        if True:
            idx = self.model.predict(item.copy(), verbose = 0)
        return idx

    def check_fast_model(self, item, compression = True):
        if compression:
            idx1 = self.fastmodel.predict_compression(item[0])
        else:
            idx1 = self.fastmodel.predict(item[0])
        return idx1


    def full_predict(self, item, unscale = False, full_predict = True):
        res = False
        time_start = time.perf_counter_ns()
        if full_predict and self.outlier_removal:
            item_set = frozenset(item)
            if self.btree_flag:
                # Other option
                # set_hash = list(item_set)#abs(hash(item_set)) % 10000000
                # np.sort(set_hash)
                set_hash = abs(hash(item_set)) % 10000000

                pos = self.b.search("==", set_hash)

                if len(pos) != 0:
                    for pos_i in pos:
                        set_i = self.original_sets[pos_i - 1]
                        if item_set.issubset(set_i):
                            idx = pos_i - 1  # set_i
                            res = True
                            break
            else:
                if self.hm_version == 2:
                    set_hash = item_set
                    res = set_hash in self.outliers_map
                    if res is True:
                        idx = int(self.outliers_map[set_hash])
                else:
                    set_hash = abs(hash(item_set)) % 10000000
                    if set_hash in self.outliers_map:
                        r = self.outliers_map[set_hash]
                        for pos_i in r:
                            set_i = self.original_sets[pos_i - 1]
                            if item_set.issubset(set_i):
                                idx = pos_i - 1
                                res = True
                                break

        time_end = time.perf_counter_ns()
        self.time_outliers += (time_end - time_start)
        # print((time_end-time_start) * 1e-6)

        if res is False:
            time_start = time.perf_counter_ns()
            set_i = transform_single_set(item, self.max_length, compression = self.compression, compression_sv_d = self.sv_d, ns = self.ns, one_hot=self.one_hot)

            if self.compression:
                idx = self.fastmodel.predict_compression(set_i)#[np.array(set_i[i]) for i in range(self.ns)])# This is with the batch [set_i[i][0] for i in range(self.ns)])
            else:
                idx = self.fastmodel.predict(set_i) # self.fastmodel.predict([i for i in item[0] if i!=0].copy())
            if unscale:
                idx = unscale_cardinality_single(idx, self.min_cardinality, self.max_cardinality)
                idx = int(idx)

            time_end = time.perf_counter_ns()
            self.time_model += (time_end - time_start)
        return idx, res


    def equality_subset_predict(self, item, equality = True):
        # time_start = time.perf_counter_ns()
        idx, true_res = self.full_predict(item, unscale=True)

        if true_res:
            return self.original_sets[idx], true_res

        time_start = time.perf_counter_ns()

        partition = error_calculator.partition(idx, min_val=0, length_of_range=100)
        error = self.errors[partition] if partition in self.errors else 1

        idx_start = idx - error - 2
        idx_end = idx + error + 2
        if idx_start < 0:
            idx_start = 0
        len_sets = len(self.original_sets)
        if idx_end >= len_sets:
            idx_end = len_sets
        idx_found = -1
        # print("idx " + str(idx) + ", error " + str(error))
        # print("idx start " + str(idx_start) + " idx end " + str(idx_end) )
        if equality:
            for i in range(idx_start, idx_end):
                if self.original_sets[i] == item:
                    idx_found = i
                    break
        else:
            item = frozenset(item)
            for i in range(idx_start, idx_end):
                if item.issubset(self.original_sets[i]):
                    idx_found = i
                    break
        time_end = time.perf_counter_ns()
        self.time_errors += (time_end - time_start)
        # print((time_end - time_start)* 1e-6)
        return idx_found, true_res



    def create_BTree(self, outliers, m = 100):
        print("Creating BTree branching factor " + str(m))
        self.b = B_Tree_own.BPlusTree(m)
        time_start = time.perf_counter_ns()
        for key in outliers:
            set_i = key
            value = outliers[key]
            # set_hash = list(set_i)#abs(hash(set_i)) % 10000000
            # np.sort(set_hash)
            set_hash = abs(hash(set_i)) % 10000000
            self.b.insert(set_hash, int(value))
        time_end = time.perf_counter_ns()
        total_time = (time_end - time_start)
        print("BTree total creation time")
        print(str((total_time) * 1e-6) + " ms")




    def predict_bf(self, item):
        set_i = transform_single_set(item, self.max_length, compression=self.compression,
                                     compression_sv_d=self.sv_d, ns=self.ns, one_hot=self.one_hot)
        if self.compression:
            idx = self.fastmodel.predict_compression(set_i)
        else:
            idx = self.fastmodel.predict(set_i)
        return idx, False
