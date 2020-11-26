#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import sys
import time

from datetime import datetime
from model import Model
from proc import get_data
from proc import get_image_batch
from progress.bar import Bar

tf_opt = sys.argv[1]
if tf_opt == 'rc':
    is_beta = False
elif tf_opt == 'beta':
    is_beta = True
else:
    print('tf option unrecognized')
    exit()

conf_opt = sys.argv[2]
lay_opt = int(sys.argv[3])
iter_opt = int(sys.argv[4])
design_opt = sys.argv[5]

if conf_opt == 'fanout':
    from config import ConfigFanout
    Config = ConfigFanout('b' + design_opt, lay_opt, iter_opt)
elif conf_opt == 'lpdc':
    from config import ConfigLPDC as Config
elif conf_opt == 'rand':
    from config import ConfigRand
    Config = ConfigRand('b' + design_opt, lay_opt, iter_opt)
elif conf_opt == 'tcad20':
    from config import ConfigTCAD20 as Config
elif conf_opt == 'tvlsi19m3':
    from config import ConfigTVLSI19M3 as Config
elif conf_opt == 'tvlsi19m5':
    from config import ConfigTVLSI19M5 as Config
elif conf_opt == 'tvlsi19m5pixel':
    from config import ConfigTVLSI19M5Pixel as Config
elif conf_opt == 'tvlsi19m6':
    from config import ConfigTVLSI19M6 as Config
else:
    print('conf option unrecognized')
    exit()


def main():
    lr = Config.learning_rate
    tf.random.set_seed(0)
    if is_beta:
        dtype = Config.dtype64
        tf.keras.backend.set_floatx('float64')
    else:
        dtype = Config.dtype32

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    data = {}
    snsc = {}
    labels = {}
    sink_name = {}
    sink_idx = {}
    img_info = {}
    max_bar = 0
    for design, layer in zip(Config.train_list + Config.cv_list, Config.train_layer + Config.cv_layer):
        path = '{0}/{1}_M{2}'.format(Config.data_path, design, layer)
        data[design], snsc[design], labels[design], sink_name[design], sink_idx[design], img_info[design] = get_data(
            path, design, layer, dtype=dtype)
        if design in Config.train_list:
            max_bar += len(sink_name[design])

    is_min_aloss = True
    is_max_cv = True
    min_aloss = 0
    max_cv = 0
    inputs = {}
    model = Model(is_beta, dtype)
    for epoch in range(0, Config.epoch):
        bar = Bar('Epoch ' + str(epoch), max=max_bar)
        aloss = []
        for design in Config.train_list:
            for s in range(0, len(sink_name[design])):
                batch_indices = np.nonzero(sink_idx[design] == s)[0]
                label = np.nonzero(labels[design][batch_indices])[0]
                if len(label) != 1:
                    continue

                inputs['data'] = data[design][batch_indices]
                inputs['image'] = get_image_batch(
                    img_info[design][batch_indices], dtype=dtype)
                model.train(inputs=inputs, labels=label, learning_rate=lr)
                aloss.append(model.lose(inputs=inputs, labels=label))
                bar.next()

        bar.finish()
        aloss = np.average(aloss)
        cv = []
        for cv_design in Config.cv_list:
            cv.append(model.stat(sink_name[cv_design], sink_idx[cv_design], data[cv_design],
                                 img_info[cv_design], labels[cv_design], snsc[cv_design]))

        cv = np.average(cv)
        print("Epoch %g : ls %f cv %f lr %f" % (epoch, aloss, cv, lr))
        if epoch % Config.show_epoch == 0 and epoch > 0:
            lr = lr*Config.lr_decay_factor

        if epoch == Config.epoch - 1 or epoch % Config.show_epoch == 0:
            is_min_aloss = True
            is_max_cv = True

        is_test = False
        if min_aloss == 0 or aloss < min_aloss:
            min_aloss = aloss
            if is_min_aloss:
                is_test = True
                is_min_aloss = False
                print(
                    "================================save for ls===================================")

        if cv > max_cv:
            max_cv = cv
            if is_max_cv:
                is_test = True
                is_max_cv = False
                print(
                    "================================save for cv===================================")

        if is_test:
            is_test = False
            model.save_weights(Config.model_path + '/tmp/epoch-' + str(epoch),
                               save_format='tf')


if __name__ == "__main__":
    main()
