from keras.utils import Sequence
import numpy as np
import math
from helper_methods import q_loss_single, inverse_transform_minmax, unscale_cardinality_single, scale_cardinality
import tensorflow as tf

class CustomSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, model):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.epoch = 0
        self.model = model

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        print("Do custom statistics")


class DataGen(Sequence):
    def __init__(self, x_set, y_set, batch_size, model, ns, compression=False, error = 0, sets = None,
                 outlierremove=False,
                 start=50, step=10, boundary=90, real_y = None, min_y = 0, max_y = 0, dataset_name = ""
                 ):
        self.x, self.y = np.array(x_set), np.array(y_set)
        self.real_y = np.array(real_y)
        self.batch_size = batch_size
        print(len(self.x.shape))
        if len(self.x.shape) == 2:
            self.indices_size = self.x.shape[0]
        else:
            self.indices_size = self.x.shape[1]
        self.indices = np.arange(self.indices_size)

        np.random.shuffle(self.indices)
        self.model = model
        self.epoch = 0
        self.steps = int(np.ceil(self.indices_size / self.batch_size))
        print("Steps: " + str(self.steps))

        if not outlierremove:
            self.start_check = 1000000
        else:
            self.start_check = start
            self.step_check = step
            self.percentile_threshold = boundary
            self.s_check = self.step_check

        self.set_outliers = []
        self.set_preds = []

        self.ns = ns
        self.compression = compression
        self.error = error
        self.sets = sets

        self.min_y = min_y
        self.max_y = max_y
        self.dataset_name = dataset_name

    def __len__(self):
        print(' len : ' + str(int(np.ceil(self.indices_size / self.batch_size))))
        return int(np.ceil(self.indices_size / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[inds]

        if not self.compression:
            batch_x = self.x[inds]
            return np.array(batch_x), np.array(batch_y)
        else:
            batch_x = self.x[:,inds]
            return [batch_x[i] for i in range(batch_x.shape[0])], np.array(batch_y)

    def on_epoch_end(self):
        if self.epoch > self.start_check:
            if self.s_check == self.step_check:
                print("Custom Sequence")
                print(self.epoch)
                if not self.compression:
                    y_pred = self.model.predict(self.x)
                else:
                    y_pred = self.model.predict([self.x[i] for i in range(self.x.shape[0])])


                len_y_pred = len(y_pred)
                errors = []
                total_err = 0
                for i in range(len_y_pred):
                    if i % 10000 == 0:
                        print(i)

                    if self.error == 0:
                        difference = q_loss_single(y_pred[i], self.y[i])
                    else:
                        difference = abs(y_pred[i] - self.y[i])
                    if math.isinf(difference):
                        print(str(y_pred[i]) + " " + str(self.y[i]))
                        exit(1)

                    errors.append(difference)
                    total_err += difference

                avg_err = total_err / len_y_pred
                # errors = np.unique(errors)
                errors_sorted = np.sort(errors.copy())
                print(errors_sorted)
                threshold = np.percentile(errors_sorted, self.percentile_threshold)
                print("Threshold " + str(threshold))
                print("Avg error " + str(avg_err))

                x = []
                y = []
                real_y = []
                sets = []
                for i, error in enumerate(errors):
                    if error < threshold:
                        if self.compression:#self.x.shape[0] == self.ns:

                            for x_s in range(self.x.shape[0]):
                                if len(x) <= x_s:
                                    x.append([])
                                x[x_s].append(self.x[x_s, i,:])

                        else:
                            x.append(self.x[i])
                        sets.append(self.sets[i])
                        real_y.append(self.real_y[i])
                        y.append(self.y[i])
                    else:
                        set_outlier = self.sets[i]#frozenset([int(x_i) for x_i in self.x[i] if x_i != 0])
                        self.set_outliers.append(set_outlier)
                        self.set_preds.append(self.real_y[i])#self.y[i])
                # if self.x.shape[0] == 2:
                if self.compression:

                    self.x = np.array(x)
                    print(np.shape(self.x))
                else:
                    self.x = np.array(x)


                self.y = np.array(y)
                print(self.y)

                self.real_y = np.array(real_y)
                self.sets = sets

                print(len(self.x))
                print(len(self.y))
                print(len(self.real_y))
                # if self.x.shape[0] == self.ns:
                if self.compression:
                    self.indices = np.arange(self.x.shape[1])
                    self.batch_size = int(np.ceil(self.x.shape[1] / self.steps))
                else:
                    self.indices = np.arange(self.x.shape[0])
                    self.batch_size = int(np.ceil(self.x.shape[0] / self.steps))
                print("The new batch size is " + str(self.batch_size))
                self.s_check = 0
            else:
                self.s_check += 1

        self.epoch += 1
        np.random.shuffle(self.indices)
        print("The epoch is " + str(self.epoch))
        print("Steps: " + str(self.steps))