from BloomFilter import BloomFilter
import math, time
import tensorflow as tf
import matplotlib.pylab as plt
from lib.utils import *
import pickle

class CLBF(object):
    def __init__(self, model, fp_rate = 0.01, model_name = ""):
        self.model = model
        self.threshold = 0.5
        self.fp_rate = float(fp_rate)
        self.model_name = model_name

    def create_complete(self, X, y):
        print("Entering inside")
        start_predict_time = time.time()
        original_preds = self.predict(X)
        print(original_preds)
        predict_time = time.time() - start_predict_time
        for threshold in [0.5]:
            self.threshold = threshold
            # self.calculate_threshold(y, original_preds)
            start_round_time = time.time()
            preds = tf.where(original_preds <= self.threshold, 0, 1)
            # print(np.sum(preds == y))
            round_time = time.time() - start_round_time
            self.analyze_predictions(y = y, preds=preds, predict_time = (predict_time + round_time))

    def read_set_elem_id(self, path):
        with open(path + "elem_id_map.pkl", 'rb') as handle:
            self.set_elem_id = pickle.load(handle)

    def read_backup_bloom_filter(self, path):
        with open(path + "backupbf.pkl", 'rb') as handle:
            self.bloom_filter = pickle.load(handle)

    def save_backup_bloom_filter(self, path):
        with open(path + "backupbf.pkl", 'wb') as file:
            pickle.dump(self.bloom_filter, file, protocol=pickle.HIGHEST_PROTOCOL)

    def predict_single(self,  max_set_size, X = None, set_i = None):
        # if it is none we are assuming that X is provided
        if set_i is not None:
            X = np.zeros((1, max_set_size))
            j = 0
            for s in set_i:
                if s in self.set_elem_id:
                    elem = self.set_elem_id[s]
                    X[0, j] = elem
                    j += 1
                else:
                    return False

        pred = self.model.predict(X)
        if pred[0] <= 0.5:
            sorted_X = np.sort(X[0])
            bf_set = ",".join([str(i) for i in sorted_X if i != 0])
            return self.bloom_filter.check(bf_set)
        else:
            return True

    # here we have only the positives
    def create_bloom_filter(self, X, y):
        print("Creating bloom filter detecting problems")
        false_negatives = []
        preds = self.model.predict(X)
        for i in range(len(y)):
            if y[i] == 1:
                if preds[i] <= self.threshold:
                    sorted_X = np.sort(X[i])
                    bf_set = ",".join([str(i) for i in sorted_X if i != 0])
                    false_negatives.append(bf_set)

        self.bloom_filter = BloomFilter(
            len(false_negatives),
            self.fp_rate / 2,
            string_digest
        )
        for fn in false_negatives:
            self.bloom_filter.add(fn)
        print("Created bloom filter")


    # sent preds need to be 1 0
    def analyze_predictions(self, X = None, y = None, preds = None, predict_time = 0.0):
        stats_file = ""
        if preds is None:
            start_predict_time = time.time()
            preds = self.predict_threshold(X)
            predict_time = time.time() - start_predict_time
            model_score = self.model.evaluate(X, y, verbose=0)
            stats_file += "Model score " + str(model_score) + "\n"

        avg_predict_time = (1.0 * predict_time) / len(preds)
        stats_file += "Predict time: " + str(predict_time) + "\n"
        stats_file += "Avg Predict time: " + str(avg_predict_time) + "\n"
        stats_file += "Predict + computation time: " + str(predict_time) + "\n"
        stats_file += "Avg Predict + computation time: " + str(avg_predict_time) + "\n"

        total_positives = 0
        total_negatives = 0
        tp = 0 # Actual: Positive, Predicted: Positive
        tn = 0 # Actual: Negative, Predicted: Negative
        fp = 0 # Actual: Negative, Predicted: Positive
        fn = 0 # Actual: Positive, Predicted: Negative
        len_y = len(y)
        for i in range(len_y):
            if i % 100000 == 0:
                print(str(i) + "/" + str(len_y))
            if y[i] == 1:
                if y[i] != preds[i]:
                    fn += 1
                else:
                    tp += 1
                total_positives += 1
            else:
                if y[i] != preds[i]:
                    fp += 1
                else:
                    tn += 1
                total_negatives += 1

        stats_accuracy = "Threshold ============ " + str(self.threshold) + " ============ "
        stats_accuracy += "Number of total positives: " + str(total_positives) + "\n"
        stats_accuracy += "Wrong from positives (FN): " + str(fn) + "\n"
        stats_accuracy += "Number of total negatives: " + str(total_negatives) + "\n"
        stats_accuracy += "Wrong from negatives (FP): " + str(fp) + "\n"
        accuracy = (1.0 * (tp + tn))/(tp + tn + fp + fn)
        precision = (1.0 * tp) / (tp + fp)
        stats_accuracy += "Accuracy: " + str(accuracy) + "\n"
        stats_accuracy += "Precision: " + str(precision) + "\n"
        print(stats_accuracy)

        stats_file += stats_accuracy
        stats_file += "Complete data" + "\n"
        stats_file += "Total data size " + str(len(preds)) + "\n"

        with open("stats/" + self.model_name + "_" + str(self.threshold) + ".txt" , "w") as f:
            f.write(stats_file)

    def predict(self, X):
        preds = self.model.predict(X)
        return preds
