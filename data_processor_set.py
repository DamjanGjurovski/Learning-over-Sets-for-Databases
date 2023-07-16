import numpy as np
import os
from tqdm import tqdm
from itertools import chain, combinations
import random as rand
import pickle
from compression import compress_elem, compress_elem_ns
import time
from custom_embedding_layer import OneHot


def all_subsets(ss, size):
    return chain(*map(lambda x: combinations(ss, x), range(1, size)))

def gen_negative_sets_posting_list_random(sets, elem_setids = None, filter_rare_terms = False, rare_terms_threshold = 3, training_data_size = 1000, max_set_size = 5):

    posting_list = dict()
    # filtering rare terms
    rare_elems = set()
    if filter_rare_terms:
        for elem in elem_setids:
            if len(elem_setids[elem]) <= rare_terms_threshold:
                rare_elems.add(elem)

    all_elems = set()
    for i, set_i in enumerate(sets):
        if filter_rare_terms:
            set_i = set([s for s in set_i if s not in rare_elems])

        for elem in set_i:
            all_elems.add(elem)
            pl_i = set()
            if elem in posting_list:
                pl_i = posting_list[elem]
            for elem_ce in set_i:
                if elem_ce != elem:
                    pl_i.add(elem_ce)
            posting_list[elem] = pl_i


    # this is only for pairs
    negative_sets = set()
    keys = list(posting_list.keys())
    # keys.sort()
    # this is only for pairs
    len_keys = len(keys)
    print("Nb of elems " + str(len_keys))

    # nb_samples_per_elem = int(training_data_size/len_keys)
    # print("Nb samples per elem " + str(nb_samples_per_elem))
    past_id_keys = set()
    # current_keys = keys.copy()
    rand.shuffle(keys)
    i = 0
    nb_keys_examined = 0
    nbTimes = 0
    flag = False
    nbRepetitions = 0
    while len(negative_sets) < training_data_size:

        elem = keys[i]
        nb_keys_examined += 1
        i += 1
        if i > len_keys:
            i = 0

        len_ns = len(negative_sets)
        if len_ns % 100 == 0:
            print(len_ns)

        pl_i = posting_list[elem]
        neg_pl_i = all_elems.copy()
        neg_pl_i.remove(elem)
        neg_pl_i = neg_pl_i.difference(pl_i)
        neg_pl_i = list(neg_pl_i)
        len_neg_pl_i = len(neg_pl_i)

        max_b = min(100, len_neg_pl_i)
        min_b = min(10, max_b) - 1
        nb_sets = rand.randint(min_b, max_b)


        for i in range(nb_sets):
            # nbRepetitions += 1
            # if nbRepetitions > 100000:
            #     flag = True
            #     print("Here " + str(nbRepetitions))
            #     break
            negative_set = set()
            negative_set.add(elem)
            size = rand.randint(0, max_set_size - 1)
            for i in range(size):
                rand_elem = rand.randint(0, len_neg_pl_i - 1)
                negative_set.add(neg_pl_i[rand_elem])
            negative_sets.add(frozenset(negative_set))
        if flag:
            break

    print(len(negative_sets))
    print("Nb of keys examined " + str(nb_keys_examined) + "/" + str(len(keys)))
    return negative_sets



# we can use the elem set ids statistics to filter out rare terms
def gen_negative_sets_posting_list(sets, elem_setids = None, filter_rare_terms = False, rare_terms_threshold = 3):
    print("Started generating negative data")
    posting_list = dict()

    # filtering rare terms
    rare_elems = set()
    if filter_rare_terms:
        for elem in elem_setids:
            if len(elem_setids[elem]) <= rare_terms_threshold:
                rare_elems.add(elem)

    all_elems = set()
    for i, set_i in enumerate(sets):
        if i % 1000 == 0:
            print(i)
        # filtering rare terms
        if filter_rare_terms:
            set_i = set([s for s in set_i if s not in rare_elems])

        for elem in set_i:
            all_elems.add(elem)
            pl_i = set()
            if elem in posting_list:
                pl_i = posting_list[elem]
            for elem_ce in set_i:
                if elem_ce != elem:
                    pl_i.add(elem_ce)
            posting_list[elem] = pl_i


    # this is only for pairs
    negative_sets = []
    keys = list(posting_list.keys())
    keys.sort()
    # this is only for pairs
    len_keys = len(keys)
    print("Nb of elems " + str(len_keys))

    all_combs = True
    for i, elem in enumerate(keys):
        if i % 1 == 0:
            print(i)
        pl_i = posting_list[elem]
        # pl_i = list(posting_list[elem])
        # pl_i.sort()

        neg_pl_i = all_elems.copy()
        neg_pl_i.remove(elem)
        neg_pl_i = neg_pl_i.difference(pl_i)
        len_neg_pl_i = len(neg_pl_i)
        print("For elem")
        # print(elem)
        print(len_neg_pl_i)
        if all_combs:
            # print("Done")
            previous_comb = []
            # this is only for pairs
            for elem_ce in neg_pl_i:
                if elem_ce > elem:
                    negative_set = set()
                    negative_set.add(elem_ce)
                    negative_set.add(elem)
                    negative_sets.append(negative_set)
                    previous_comb.append(negative_set)
        else:
            nb_sets = rand.randint(1, 100)
            for i in range(nb_sets):
                negative_set = set()
                negative_set.add(elem)
                size = rand.randint(2, 5)
                for i in range(size):
                    rand_elem = rand.randint(1, len_neg_pl_i)
                    negative_set.add(neg_pl_i[rand_elem])
                print(negative_set)

    print(len(negative_sets))
    return posting_list




def gen_negative_sets(elem_setids, num_sets, max_set_length):
    negative_sets = []
    elems = list(elem_setids.keys())
    # print(elems)
    # exit(1)
    while len(negative_sets) < num_sets:
        print(len(negative_sets))
        length = np.random.randint(2, max_set_length + 1)
        negative_set = set()
        setids = set()
        # for l in range(length):
        nb_times_before_break = 100
        while len(negative_set) < length and nb_times_before_break > 0:
            nb_times_before_break -= 1
            # print(nb_times_before_break)
            # print(length)
            elem = np.random.choice(elems)
            curr_elem_setids = elem_setids[elem]
            overlap = setids.intersection(curr_elem_setids)
            # if they do not belong to the same group
            if len(overlap) == 0:
                setids.update(curr_elem_setids)
                # print(setids)
                negative_set.add(elem)
        if len(negative_set) != 1:
            negative_sets.append(negative_set)
    return negative_sets


def gen_negative_sets_other(elem_setids, iterations):
    '''

    :param elem_setids: dictionary containing the element and the ids of the sets where the elements appear
    :param iterations: how many iterations we will have, if we have 1 it means we will create non-existent pairs
    :return:
    '''
    negative_sets = dict()
    previous_sets = dict()

    elems2 = list(elem_setids.keys())
    len_elems2 = len(elems2)
    elem_setids2 = elem_setids

    t = 0
    for t in range(iterations):
        if t == 0:
            elems1 = elems2
            elem_setids1 = elem_setids
        else:
            elems1 = list(previous_sets.keys())
            elem_setids1 = previous_sets
            previous_sets = dict()

        len_elems1 = len(elems1)


        for i in range(len_elems1):
            if i % 10 == 0:
                print(str(i) + "/" + str(len_elems1) + "======")
            if len_elems1 < 1000:
                continue
            elem_i = elems1[i]
            elem_i_ids = set(elem_setids1[elem_i])
            start = (i + 1) if (t == 0) else 0
            for j in range(start, len_elems2):
                if j % 10000 == 0:
                    print(str(j) + "/" + str(len_elems2))
                elem_j = elems2[j]
                elem_j_ids = set(elem_setids2[elem_j])
                if t == 0:
                    if elem_i == elem_j:
                        continue
                else:
                    if elem_j in elem_i:
                        print("Continue")
                        continue
                intersection_ids = elem_i_ids.intersection(elem_j_ids)
                if len(intersection_ids) == 0:
                    combination_ids = set()
                    combination_ids.update(elem_i_ids)
                    combination_ids.update(elem_j_ids)
                    combination_elems = set()
                    # if element it needs to be add otherwise a set
                    if t == 0:
                        combination_elems.add(elem_i)
                    else:
                        combination_elems.update(elem_i)
                    combination_elems.add(elem_j)
                    negative_sets[frozenset(combination_elems)] = combination_ids
                    previous_sets[frozenset(combination_elems)] = combination_ids
    return negative_sets

def create_bf_training_data(sets = None, elem_setids = None, path = None, dir_path = None, max_set_size = 10, generate_positives = True, num_sets = 10000000):
    if sets is None:
        sets, elem_setids = read_sets(path, create_map=True)


    if generate_positives:
        positive_sets = create_positive_sets_bf(sets, max_set_size = max_set_size)
        num_sets = len(positive_sets)
        print("Positive sets length: " + str(num_sets))
        store_sets(positive_sets, dir_path + "bf_positive.txt")
    # else:
    #     num_sets = 1000000


    negative_sets = gen_negative_sets_posting_list_random(sets, elem_setids = elem_setids, training_data_size = num_sets , max_set_size = 5)#max_set_size)
    print("Negative sets length: " + str(len(negative_sets)))
    if generate_positives:
        store_sets(negative_sets, dir_path + "bf_negative.txt")
    else:
        store_sets(negative_sets, dir_path + "bf_negative_" + str(num_sets) + ".txt")




def read_training_data():
    directory = "example_data_" + str(num_sets) + "_min_" + str(min_elem) + "_max_" + str(max_elem)
    # Parent Directory path
    parent_dir = "data/"
    # Path
    full_path = parent_dir + directory + "/"
    positive_sets, _ = read_sets(full_path + "positive.txt")
    negative_sets, _ = read_sets(full_path + "negative.txt")
    return positive_sets, negative_sets

def store_sets(sets, path):
    with open(path, "w", encoding="utf-8") as f_w:
        for set in sets:
            f_w.write(','.join(str(s) for s in set) + "\n")

def read_sets(path, delimiter = ",", cast = True, create_map = False):
    sets = []
    elem_setids = dict()

    max_set_size = 0
    max_elem = 0
    with open(path, "r", encoding="utf8") as f_r:
        for line_nb, line in enumerate(f_r):
            # print(line)
            if line_nb == 1000000 and "negative" in path:
                print("breaking ")
                print(path)
                break
            line = line.replace("\n", "")
            if line_nb % 1000 == 0:
                print(line_nb)

            if cast:
                s = set([int(x) for x in line.split(delimiter)])
            else:
                s = set([x for x in line.split(delimiter)])

            if cast:
                max_set_size = max(max_set_size, len(s))
                max_elem = max(max_elem, max(s))
                sets.append(s)

            if create_map:
                for elem in s:
                    setids = []
                    if elem in elem_setids:
                        setids = elem_setids[elem]
                    setids.append(line_nb)
                    elem_setids[elem] = setids

    print("The max set size is " + str(max_set_size))
    print("The max elem is " + str(max_elem))
    return sets, elem_setids

def read_sets_cardinality(path):
    print("Starting to read cardinality ...")
    sets = []
    cards = []
    with open(path, "r", encoding="utf8") as f_r:
        for l_nb, line in enumerate(f_r):
            # if l_nb == 4000000:
            #     break
            if l_nb % 1000 == 0:
                print(l_nb)
            set_i, card_i = line.split(":")
            s = frozenset([int(x) for x in set_i.split(",")])# THIS WAS CHANGED FROM SET TO FROZENSET
            card_i = int(card_i)
            # if len(s) < 8:
            sets.append(s)
            cards.append(card_i)
            # else:
            #     print("Larger than 8 please check")
    return sets, cards

def read_transform_sets():
    positive_sets, negative_sets = read_training_data()
    max_train_length = 10
    num_train_examples = len(positive_sets) + len(negative_sets)
    print(num_train_examples)
    X = np.zeros((num_train_examples, max_train_length))
    sum_X = np.zeros((num_train_examples))
    ix = 0
    for i in tqdm(range(len(positive_sets)), desc='Generating train examples: '):
        set_i = list(positive_sets[i])
        n = len(set_i)
        for j in range(1, n + 1):
            X[ix, -j] = set_i[j-1]
        sum_X[ix] = 1
        ix += 1

    for i in tqdm(range(len(negative_sets)), desc='Generating train examples: '):
        set_i = list(negative_sets[i])
        n = len(set_i)
        for j in range(1, n + 1):
            X[ix, -j] = set_i[j-1]
        sum_X[ix] = 0
        ix += 1

    return X, sum_X

def calculate_store_cardinality(sets, path, max_set_size = 5):
    subset_cardinality = dict()
    for set_id, set in enumerate(sets):
        if set_id % 1000 == 0:
            print(set_id)

        all_combs = all_subsets(set, max_set_size)
        for comb in all_combs:
            # we need to sort it
            sort_comb = np.sort(comb)
            comb_str = ",".join([str(c) for c in sort_comb])
            card = 0
            if comb_str in subset_cardinality:
                card = subset_cardinality[comb_str]
            card += 1
            subset_cardinality[comb_str] =  card
    # here store the cardinality
    store_map(path + "cardinality_max_size" + str(max_set_size) + ".txt", subset_cardinality)
    return subset_cardinality

# this is for the secondary index
def calculate_store_min_max_boundaries(sets, path, max_set_size):
    min_boundary = dict()
    max_boundary = dict()
    for set_id, set in enumerate(sets):
        if set_id % 1000 == 0:
            print(set_id)
        # If you want to limit larger sets
        # if len(set)>8:
        #     continue
        all_combs = all_subsets(set, max_set_size)
        for comb in all_combs:
            # we need to sort it
            fs_comb = frozenset(comb)
            max_boundary[fs_comb] = set_id + 1
            if fs_comb not in min_boundary:
                min_boundary[fs_comb] = set_id + 1

    store_map(path + "min_boundary_"+str(max_set_size)+".txt", min_boundary)
    store_map(path + "max_boundary_"+str(max_set_size)+".txt", max_boundary)
    return min_boundary, max_boundary


def create_positive_sets_bf(sets, max_set_size = 5):
    positive_sets = set()
    for set_id, set_i in enumerate(sets):
        if set_id % 1000 == 0:
            print(set_id)
        # TODO CHANGE THIS
        all_combs = all_subsets(set_i, max_set_size)
        for comb in all_combs:
            positive_sets.add(frozenset(comb))
    return positive_sets


def transform_single_set_batch(set_i, max_train_length, compression = False, compression_sv_d = 0, ns = 2, one_hot = True):
    num_train_examples = 1
    if not compression:
        X = np.zeros((num_train_examples, max_train_length))
    else:
        X_compressed = []
        for i in range(ns):
            X_i = np.full((num_train_examples, max_train_length), -1 if one_hot else 0)
            X_compressed.append(X_i)
    ix = 0
    n = len(set_i)
    for j in range(1, n + 1):
        number = set_i[j - 1]
        if not compression:
            X[ix, -j] = number
        else:
            compressed_elems = compress_elem_ns(number, compression_sv_d, ns, one_hot=one_hot)
            for xc_i in range(len(compressed_elems)):
                X_compressed[xc_i][ix, -j] = compressed_elems[xc_i]

    if compression:
        return X_compressed
    else:
        return X


def transform_single_set(set_i, max_train_length, compression = False, compression_sv_d = 0, ns = 2, one_hot = True):
    if not compression:
        X = np.zeros(max_train_length)
        X[:set_i.shape[0]] = set_i
        return X
    else:

        X_compressed= np.full((ns, max_train_length), -1 if one_hot else 0)

        # VARIANT 1
        # elems = np.array(compress_elem_ns(set_i, compression_sv_d, ns, one_hot=one_hot))
        # X_compressed[:elems.shape[0], :elems.shape[1]] = elems
        # VARIANT 1

        # VARIANT 2
        n = len(set_i)
        for j in range(1, n + 1):
            number = set_i[j - 1]
            compressed_elems = compress_elem_ns(number, compression_sv_d, ns, one_hot=one_hot)
            for xc_i in range(len(compressed_elems)):
                X_compressed[xc_i][-j] = compressed_elems[xc_i]
                    # x_dim_max[xc_i] = max(x_dim_max[xc_i], compressed_elems[xc_i])
        # print(X_compressed)
        # VARIANT 2
        return X_compressed





def read_transform_sets_cardinality(path, max_train_length = 10, compression = False, compression_sv_d = 0, ns = 2, one_hot = True):
    print("We are reading the set cardinalities or the respective index positions/")
    old_compression = True
    sets, cardinality = read_sets_cardinality(path)
    set_card_map = dict()
    for i in range(len(sets)):
        set_card_map[frozenset(sets[i])] = int(cardinality[i])

    num_train_examples = len(sets)
    x_dim_max = [0 for i in range(ns)]
    if not compression:
        X = np.zeros((num_train_examples, max_train_length))
    else:
        if not old_compression:
            X_compressed_full = np.full((num_train_examples, max_train_length, ns), -1 if one_hot else 0)
        else:
            X_compressed = []
            for i in range(ns):
                X_i = np.full((num_train_examples, max_train_length), -1 if one_hot else 0)
                X_compressed.append(X_i)


    sum_X = np.zeros((num_train_examples))
    ix = 0
    print("TQDM")
    for i in tqdm(range(len(sets)), desc='Generating train examples: '):
        set_i = list(sets[i])
        n = len(set_i)
        for j in range(1, n + 1):
            number = set_i[j - 1]
            if not compression:
                X[ix, -j] = number
            else:
                compressed_elems = compress_elem_ns(number, compression_sv_d, ns, one_hot=one_hot)
                if old_compression:
                    for xc_i in range(len(compressed_elems)):
                        X_compressed[xc_i][ix, -j] = compressed_elems[xc_i]
                        x_dim_max[xc_i] = max(x_dim_max[xc_i], compressed_elems[xc_i])
                else:
                    for c_e_i, c_e in enumerate(compressed_elems):
                        X_compressed_full[ix, -j, c_e_i] = c_e


        sum_X[ix] = cardinality[i]
        ix += 1
    if not compression:
        return X, sum_X, x_dim_max, sets, set_card_map
    else:
        # this is needed if we have the one-hot encoding as a part of the model
        if old_compression:
            return X_compressed, sum_X, x_dim_max, sets, set_card_map
        else:
            max_elem_compressed_dim = compression_sv_d + 1
            X_compressed_full = transform_compression(X_compressed_full, max_elem_compressed_dim, compression_sv_d, ns, num_train_examples,
                                                 max_set_length=max_train_length)
            return X_compressed_full, sum_X, x_dim_max, sets, set_card_map



def transform_compression_old(X_compressed, max_elem_compressed_dim, sv_d, ns, num_examples, max_set_length):
    X_compressed = (OneHot(input_dim=max_elem_compressed_dim, input_length=max_set_length)(X_compressed))
    X_compressed = np.array(X_compressed).reshape(
        (num_examples, max_set_length, ns * max_elem_compressed_dim))
    return X_compressed

# added
def transform_compression(X_compressed, max_elem_compressed_dim, sv_d, ns, num_examples, max_set_length):
    option = 0
    if option == 0:
        X_compressed_full = []
        for i, set_i in enumerate(X_compressed):
            if i % 1000 == 0:
                print(i)
            if i==1000:
                break
            set_c_o_h = []
            for elems_compr in set_i:
                elems_compr_o_h = get_one_hot_tf(elems_compr, max_elem_compressed_dim)#.reshape(-1)
                set_c_o_h.append(elems_compr_o_h)

            X_compressed_full.append(set_c_o_h)

        X_compressed_full = np.array(X_compressed_full).reshape((num_examples, max_set_length, ns * max_elem_compressed_dim))
        return X_compressed_full
    elif option == 1:
        X_compressed = (OneHot(input_dim=max_elem_compressed_dim, input_length=max_set_length)(X_compressed))
        X_compressed = np.array(X_compressed).reshape((num_examples, max_set_length, ns * max_elem_compressed_dim))
        return X_compressed
    else:
        nb_splits = 10
        Z = np.array_split(X_compressed, nb_splits)
        X_compressed_full = None
        for i in range(nb_splits):
            X_c_i = (OneHot(input_dim=max_elem_compressed_dim, input_length=max_set_length)(Z[i]))
            print(np.shape(X_c_i))
            if X_compressed_full is None:
                X_compressed_full = X_c_i.numpy()
            else:
                print(type(X_c_i))
                X_compressed_full = np.concatenate([X_compressed_full, X_c_i.numpy()])
        X_compressed_full = np.array(X_compressed_full).reshape((num_examples, max_set_length, ns * max_elem_compressed_dim))
        return X_compressed_full




def store_map(path, subset_cardinality):
    with open(path, "w", encoding="utf8") as f_w:
        for subset in subset_cardinality:
            if isinstance(subset, str):
                f_w.write(subset + ":" + str(subset_cardinality[subset]) + "\n")
            else:
                f_w.write(",".join([str(s) for s in subset]) + ":" + str(subset_cardinality[subset]) + "\n")


def all_subsets(ss, size):
    return chain(*map(lambda x: combinations(ss, x), range(1, size + 1)))

def read_transform_sets_to_int(path, delimiter = ",", dir_path = None):
    '''
    Checked. The method tranforms the keys into ids
    :param path: location of the sets
    :param delimiter: delimiter of the sets
    :return:
    '''
    sets = []
    elem_id_map = dict()
    id = 1
    max_set_size = 1
    with open(path, "r", encoding="utf8") as f_r:
        for line_nb, line in enumerate(f_r):
            line = line.replace("\n", "")
            set_i = []
            # print(line.split("|"))
            for x in line.split(delimiter):
                if "ger_shopamenity" in path:
                    if ":" in x:
                        continue
                curr_id = 0
                if x in elem_id_map:
                    curr_id = elem_id_map[x]
                else:
                    elem_id_map[x] = id
                    curr_id = id
                    id += 1
                set_i.append(curr_id)
            sets.append(set_i)
            max_set_size = max(max_set_size, len(set_i))
    save_dict = True
    if save_dict:
        with open(dir_path + 'elem_id_map.pkl', 'wb') as f:
            pickle.dump(elem_id_map, f)
    return sets

def create_set_elemids_dict(sets):
    set_elemids = dict()
    for id, set in enumerate(sets):
        for s in set:
            ids = []
            if s in set_elemids:
                ids = set_elemids[s]
            ids.append(id)
            set_elemids[s] = ids
    return set_elemids

def create_msmmq_data_positive(set_elemids, write_file):
    f_w = open(write_file + "_msmmq_positive.csv", "w")
    f_w.write("elementid, setid\n")
    for elem in set_elemids:
        set_ids = set_elemids[elem]
        if len(set_ids) < 10000:
            continue
        for s_id in set_ids:
            f_w.write(str(elem) + "," + str(s_id + 1) + "\n")
    print("Created positive pairs")

def create_msmmq_data_negative(set_elemids, min_set_id, max_set_id, write_file):
    f_w = open(write_file + "_msmmq_negative.csv", "w")
    f_w.write("elementid, setid\n")
    for elem in set_elemids:
        set_ids = set_elemids[elem]
        # if len(set_ids) < 10000:
        #     continue
        for s_id in range(min_set_id, max_set_id + 1):
            if s_id not in set_ids:
                f_w.write(str(elem) + "," + str(s_id + 1) + "\n")
    print("Created negative pairs")

def read_training_data_bf(positive_path, negative_path):
    if positive_path != "":
        positive_sets, _ = read_sets(positive_path)
    else:
        positive_sets = []

    if negative_path != "":
        negative_sets, _ = read_sets(negative_path)
    else:
        negative_sets = []
    return positive_sets, negative_sets


def transform_sets_individual(sets, max_set_size = 10, compression = False, compression_sv_d = 0, ns = 2):
    num_sets = len(sets)
    if not compression:
        X = np.zeros((num_sets, max_set_size))
    else:
        X_compressed = []
        for i in range(ns):
            X_i = np.full((num_sets, max_set_size), -1)
            X_compressed.append(X_i)
        X_compressed_full = np.full((num_sets, max_set_size, ns), -1)

    ix = 0
    for i in tqdm(range(len(sets)), desc='Generating train examples: '):
        set_i = list(sets[i])
        n = len(set_i)
        for j in range(1, n + 1):
            number = set_i[j-1]
            if not compression:
                X[ix, -j] = number
            else:
                compressed_elems = compress_elem_ns(number, compression_sv_d, ns)
                # added
                for c_e_i, c_e in enumerate(compressed_elems):
                    X_compressed_full[ix, -j, c_e_i] = c_e

                    X_compressed[c_e_i][ix, -j] = c_e

        ix += 1
    old_variant = True
    if not compression:
        return X
    else:
        if old_variant:
            return X_compressed
        max_elem_compressed_dim = compression_sv_d + 1
        X_compressed_full = transform_compression(X_compressed_full, max_elem_compressed_dim, compression_sv_d, ns,
                                                  num_sets,
                                                  max_set_length=max_set_size)
        return X_compressed_full

def read_transform_sets_bf(positive_path, negative_path, max_set_size = 10, compression = False, compression_sv_d = 0, ns = 2, one_hot = True):
    positive_sets, negative_sets = read_training_data_bf(positive_path, negative_path)
    num_train_examples = len(positive_sets) + len(negative_sets)
    print(num_train_examples)
    time_start = time.time()

    old_compression = True
    if not compression:
        X = np.zeros((num_train_examples, max_set_size))
    else:
        if not old_compression:
            # added
            X_compressed_full = np.full((num_train_examples, max_set_size, ns), -1 if one_hot else 0)
            # added
        else:
            X_compressed = []
            for i in range(ns):
                X_i = np.full((num_train_examples, max_set_size), -1 if one_hot else 0)
                #X_i = np.zeros((num_train_examples, max_train_length))
                X_compressed.append(X_i)

    x_dim_max = [0 for i in range(ns)]


    sum_X = np.zeros((num_train_examples))
    ix = 0
    for i in tqdm(range(len(positive_sets)), desc='Generating train examples: '):
        set_i = list(positive_sets[i])
        n = len(set_i)
        for j in range(1, n + 1):
            number = set_i[j-1]

            if not compression:
                X[ix, -j] = number
            else:
                if old_compression:
                    compressed_elems = compress_elem_ns(number, compression_sv_d, ns, one_hot=one_hot)
                    for xc_i in range(len(compressed_elems)):
                        X_compressed[xc_i][ix, -j] = compressed_elems[xc_i]
                        x_dim_max[xc_i] = max(x_dim_max[xc_i], compressed_elems[xc_i])
                else:
                    # added
                    for c_e_i, c_e in enumerate(compressed_elems):
                        X_compressed_full[ix, -j, c_e_i] = c_e

        sum_X[ix] = 1
        ix += 1

    for i in tqdm(range(len(negative_sets)), desc='Generating train examples: '):
        set_i = list(negative_sets[i])
        n = len(set_i)
        for j in range(1, n + 1):
            number = set_i[j - 1]

            if not compression:
                X[ix, -j] = number
            else:
                if old_compression:
                    compressed_elems = compress_elem_ns(number, compression_sv_d, ns, one_hot=one_hot)
                    for xc_i in range(len(compressed_elems)):
                        X_compressed[xc_i][ix, -j] = compressed_elems[xc_i]
                        x_dim_max[xc_i] = max(x_dim_max[xc_i], compressed_elems[xc_i])
                else:
                    # added
                    for c_e_i, c_e in enumerate(compressed_elems):
                        X_compressed_full[ix, -j, c_e_i] = c_e

        sum_X[ix] = 0
        ix += 1
    time_end = (time.time() - time_start)
    print("Num of encoded samples " + str(num_train_examples))
    print("Time for encoding and compression for num of encoded samples " + str(time_end/1000) + "ms")

    if not compression:
        return X, sum_X, x_dim_max
    else:
        if old_compression:
            return X_compressed, sum_X, x_dim_max
        else:

            max_elem_compressed_dim = compression_sv_d + 1
            X_compressed_full = transform_compression(X_compressed_full, max_elem_compressed_dim, compression_sv_d, ns, num_train_examples,
                                                 max_set_length=max_set_size)
            return X_compressed_full, sum_X, x_dim_max

