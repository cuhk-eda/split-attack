import numpy as np
import pandas as pd
import random
import tensorflow as tf

from proc import get_image_batch
from proc import get_image_batch_sep


class ResLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation, name, dtype):
        super(ResLayer, self).__init__(name=name, dtype=dtype)
        self.units = units
        self.activation = activation
        self.prefix = name
        self.dense_layers = []
        self.dense_layers.append(tf.keras.layers.Dense(
            self.units, activation=self.activation, name='{0}_dense_1'.format(self.prefix), dtype=self.dtype))
        self.dense_layers.append(tf.keras.layers.Dense(
            self.units, activation=self.activation, name='{0}_dense_2'.format(self.prefix), dtype=self.dtype))
        self.dense_layers.append(tf.keras.layers.Dense(
            self.units, name='{0}_dense_3'.format(self.prefix), dtype=self.dtype))

    @tf.function
    def call(self, inputs):
        x = [inputs]
        for layer in self.dense_layers:
            x.append(layer(x[-1]))

        return self.activation(x[0] + x[-1])


class Model(tf.keras.Model):
    def __init__(self, is_beta, dtype):
        super(Model, self).__init__(name='model', dtype=dtype)
        self.is_beta = is_beta
        self.units = 128
        self.data_seq = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.units, activation=self.activation, name='data_dense_1', dtype=self.dtype)])
        for i in range(0, 4):
            self.data_seq.add(
                ResLayer(units=self.units, activation=self.activation, name='data_res_{0}'.format(i + 1), dtype=self.dtype))

        self.image_seq = tf.keras.Sequential()
        for i in range(0, 12):
            filters = 16 * (2 ** (i // 3))
            strides = (1, 1)
            if i and i % 3 == 0:
                strides = (3, 3)

            padding = 'same'
            if i == 3 or i == 6:
                padding = 'valid'

            self.image_seq.add(tf.keras.layers.Conv2D(
                filters, [3, 3], strides=strides, padding=padding, activation=self.activation, name='image_conv_{0}'.format(i + 1),
                dtype=self.dtype))

        if self.is_beta:
            self.flatten = tf.keras.layers.Flatten()
            self.pair_seq = tf.keras.Sequential(
                [tf.keras.layers.Dense(256, activation=self.activation, name='pair_dense_1', dtype=self.dtype),
                 tf.keras.layers.Dense(self.units, activation=self.activation, name='pair_dense_2', dtype=self.dtype)])
        else:
            self.image_seq.add(tf.keras.layers.Flatten())
            self.image_seq.add(tf.keras.layers.Dense(
                256, activation=self.activation, name='image_dense_1', dtype=self.dtype))
            self.image_seq.add(tf.keras.layers.Dense(
                self.units, activation=self.activation, name='image_dense_2', dtype=self.dtype))
            self.pair_dense = tf.keras.layers.Dense(
                self.units, activation=self.activation, name='pair_dense_1', dtype=self.dtype)

        self.merge_seq = tf.keras.Sequential([tf.keras.layers.Dense(
            self.units, activation=self.activation, name='merge_dense_1', dtype=self.dtype)])
        for i in range(0, 3):
            self.merge_seq.add(ResLayer(units=self.units, activation=self.activation,
                                        name='merge_res_{0}'.format(i + 1), dtype=dtype))

        self.merge_seq.add(tf.keras.layers.Dense(
            32, activation=self.activation, name='merge_dense_2', dtype=self.dtype))
        self.merge_seq.add(tf.keras.layers.Dense(
            1, activation=self.activation, name='merge_dense_3', dtype=self.dtype))

        self.adam = tf.keras.optimizers.Adam(beta_1=0.99)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        random.seed(0)

    @tf.function
    def call(self, inputs):
        data = self.data_seq(inputs['data'])
        image = self.image_seq(inputs['image'])
        if self.is_beta:
            image = self.flatten(image)

        snk_image = tf.tile(image[0:1], [tf.shape(image)[0] - 1, 1])
        image = tf.concat([snk_image, image[1:]], axis=-1)
        if self.is_beta:
            image = self.pair_seq(image)
        else:
            image = self.pair_dense(image)

        data = tf.keras.layers.concatenate([data, image])
        data = self.merge_seq(data)
        return tf.reshape(data, [1, -1])

    def activation(self, features):
        return tf.nn.leaky_relu(features, alpha=0.01, name=None)

    @tf.function
    def lose(self, inputs, labels):
        logits = self(inputs=inputs)
        return self.loss(labels, logits)

    @tf.function
    def train(self, inputs, labels, learning_rate):
        with tf.GradientTape() as tape:
            pred_loss = self.lose(inputs=inputs, labels=labels)

        gradients = tape.gradient(pred_loss, self.trainable_variables)
        self.adam.learning_rate = learning_rate
        self.adam.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def gradient(self, inputs, labels):
        with tf.GradientTape() as tape:
            tape.watch(inputs['image'])
            pred_loss = self.lose(inputs=inputs, labels=labels)

        return tf.math.argmax(tf.reshape(tape.gradient(pred_loss, inputs['image']), [-1, 99 * 99, 3]), 1)

    def stat(self, names, idces, datas, imgs, labels, snscs, is_adv=False):
        cor_snsc = 0
        all_snsc = 0
        inputs = {}
        for s in range(0, len(names)):
            batch_indices = np.nonzero(idces == s)[0]
            snsc = snscs[batch_indices][0, 0]
            all_snsc += snsc
            label = labels[batch_indices][:, 0]
            if not np.any(label):
                continue

            inputs['data'] = datas[batch_indices]
            inputs['image'] = get_image_batch(imgs[batch_indices], self.dtype)
            if is_adv:
                label_idx = np.nonzero(label)
                num_image = inputs['image'].shape[0]
                num_adv = 50
                p = [0] * num_adv
                x = []
                for i in range(0, num_adv):
                    x.append((random.randrange(0, num_image),
                              random.randrange(0, 99), random.randrange(0, 99), random.randrange(0, 3)))

                max_i = 0
                for _ in range(0, 50):
                    for i in range(0, num_adv):
                        r0 = random.randrange(0, num_adv)
                        r1 = random.randrange(0, num_adv)
                        image_idx = max(
                            0, min(num_image - 1, int(x[i][0] + 0.5 * (x[r0][0] - x[r1][0]))))
                        v = max(
                            0, min(98, int(x[i][1] + 0.5 * (x[r0][1] - x[r1][1]))))
                        h = max(
                            0, min(98, int(x[i][2] + 0.5 * (x[r0][2] - x[r1][2]))))
                        d = max(
                            0, min(2, int(x[i][3] + 0.5 * (x[r0][3] - x[r1][3]))))
                        t = (image_idx, v, h, d)
                        pixel = inputs['image'][t]
                        inputs['image'][x[i]] = 255
                        pred_loss = self.lose(inputs=inputs, labels=label_idx)
                        inputs['image'][t] = pixel
                        if pred_loss > p[i]:
                            p[i] = pred_loss
                            x[i] = t

                        if p[i] > p[max_i]:
                            max_i = i

                inputs['image'][x[max_i]] = 255

            prob = self(inputs=inputs)[0]
            pred_label = np.argmax(prob)

            if label[pred_label]:
                cor_snsc += snsc

        return cor_snsc / all_snsc

    def stat_sep(self, drv_df, snk_df, snk_nets, path):
        net_indicator = ['DESIGN', 'PARENT', 'NAME', 'SINK_COUNT']
        drv_nets = drv_df.df[net_indicator].drop_duplicates(
        ).sort_values(by=['NAME']).reset_index(drop=True)
        probes = np.zeros((0, 7))
        cor_snk = 0
        all_snk = 0
        zero = self(inputs={'data': np.zeros((1, 30), dtype=self.dtype),
                            'image': np.zeros((2, 99, 99, 3), dtype=self.dtype)})[0, 0]
        for snk_idx, snk_name, snk_vias in snk_df:
            data, drv_imgs, snk_imgs, label = get_image_batch_sep(
                drv_df, snk_vias, drv_nets, snk_name, drv_size=0, dtype=self.dtype)
            drv_size = data.shape[0]
            probe = np.zeros((drv_size, 7))
            probe[:, 0] = snk_idx
            probe[:, 1:3] = data[:, 2:4]
            probe[:, 3:5] = data[:, 6:8]
            all_snk += 1
            if label >= 0:
                probe[label, 6] = 1
                cor_snk += 1

            batch_size = 1024
            for i in range(0, drv_size, batch_size):
                j = min(i + batch_size, drv_size)
                image = np.concatenate((snk_imgs, drv_imgs[i:j]))
                probe[i:j,
                      5] += (self(inputs={'data': data[i:j], 'image': image})[0] - zero)

            probes = np.concatenate((probes, probe))

        design = snk_nets.loc[0, 'DESIGN']
        pd.DataFrame(probes,
                     columns=['IDX', 'UNSIGNED_ABSOLUTE_DIST_X',
                              'UNSIGNED_ABSOLUTE_DIST_Y',
                              'UNSIGNED_RELATIVE_DIST_X',
                              'UNSIGNED_RELATIVE_DIST_Y', 'PROBE', 'LABEL'
                              ]).to_csv('{0}/{1}.prb.nrm.csv'.format(path,
                                                                     design),
                                        index=False)

        return cor_snk / all_snk
