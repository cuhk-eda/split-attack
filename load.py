#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import sys
import time

from datetime import datetime
from model import Model
from proc import get_data
from proc import get_data_sep
from proc import get_image_batch

epoch_opt = int(sys.argv[1])

sink_opt = sys.argv[2]
if sink_opt == 'all':
    is_sel = False
elif sink_opt == 'sel':
    is_sel = True
else:
    print('sink option unrecognized')
    exit()

tf_opt = sys.argv[3]
if tf_opt == 'rc':
    is_beta = False
elif tf_opt == 'beta':
    is_beta = True
else:
    print('tf option unrecognized')
    exit()

conf_opt = sys.argv[4]
lay_opt = int(sys.argv[5])
iter_opt = int(sys.argv[6])
design_opt = sys.argv[7]

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
    jv = {'b11': 0.666667, 'c1355': 0.896104, 'c432': 0.767442,
          'c2670': 0.548544, 'b13': 0.420455, 'c880': 0.714286,
          'c1908': 0.944444, 'b7': 0.556522, 'c7552': 0.503378,
          'c6288': 0.631579, 'c3540': 0.548673, 'c5315': 0.522034,
          'b14': 0.303259, 'b15_1': 0.264155}

    tf.random.set_seed(0)
    if is_beta:
        dtype = Config.dtype64
        tf.keras.backend.set_floatx('float64')
    else:
        dtype = Config.dtype32

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    data = {}
    snsc = {}
    labels = {}
    sink_name = {}
    sink_idx = {}
    img_info = {}
    for design, layer in zip(Config.cv_list, Config.cv_layer):
        path = '{0}/{1}_M{2}'.format(Config.data_path, design, layer)
        data[design], snsc[design], labels[design], sink_name[design], sink_idx[design], img_info[design] = get_data(
            path, design, layer, dtype=dtype)

    inputs = {}
    model = Model(is_beta, dtype)
    design = Config.cv_list[0]
    for s in range(0, len(sink_name[design])):
        batch_indices = np.nonzero(sink_idx[design] == s)[0]
        label = np.nonzero(labels[design][batch_indices])[0]
        if len(label) == 0:
            continue

        inputs['data'] = data[design][batch_indices]
        inputs['image'] = get_image_batch(
            img_info[design][batch_indices], dtype=dtype)
        model.train(inputs=inputs, labels=label,
                    learning_rate=Config.learning_rate)
        break

    if is_sel:
        model.load_weights(
            '{0}/epoch-{1}'.format(Config.model_path, epoch_opt))
    else:
        model.load_weights(
            '{0}/tmp/epoch-{1}'.format(Config.model_path, epoch_opt))

    cv = []
    for design in Config.cv_list:
        cv.append(model.stat(sink_name[design], sink_idx[design],
                             data[design], img_info[design], labels[design], snsc[design]))

    print("Epoch %g : ls    cv %f" % (epoch_opt, np.average(cv)))
    print("================================test for ld===================================")

    score = 0
    num_designs = 0
    for design, layer in zip(Config.test_list, Config.test_layer):
        start_time = time.time()
        path = '{0}/{1}_M{2}'.format(Config.data_path, design, layer)
        if is_sel:
            drv_df, snk_df, snk_nets = get_data_sep(
                path, layer, dtype=dtype)
            s = model.stat_sep(drv_df, snk_df, snk_nets, Config.model_path)
            print('{0}\t{1}\t{2}\t{3}'.format(
                design, snk_nets.shape[0], s, time.time() - start_time))
        else:
            data, snsc, labels, sink_name, sink_idx, img_info = get_data(
                path, design, layer, dtype=dtype, clean_img=True)
            s = model.stat(sink_name, sink_idx, data, img_info,
                           labels, snsc, is_adv=False)

            if design in jv:
                print('{0}\t{1}\t{2}\t{3}'.format(
                    design, s, jv[design], time.time() - start_time))
                score += s / jv[design]
                num_designs += 1
            else:
                print('{0}\t{1}\t{2}'.format(
                    design, s, time.time() - start_time))

    if not is_sel:
        print(score / num_designs)

    print(
        "===========================++++++++++++++++===================================")


if __name__ == "__main__":
    main()
