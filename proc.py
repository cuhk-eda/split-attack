import imageio
import numpy as np
import pandas as pd
import sys

png_dict = {}


class DFIterator:
    def next(self):
        self.idx += 1
        self.begin = self.end
        if self.begin < self.len:
            self.name = self.df.at[self.begin, 'NAME']

        for i in range(self.begin, self.len + 1):
            if i == self.len or self.df.at[i, 'NAME'] != self.name:
                self.end = i
                break

    def __init__(self, df):
        self.df = df
        self.len = self.df.shape[0]
        self.idx = -1
        self.end = 0
        self.next()

    def __iter__(self):
        return self

    def __next__(self):
        if self.begin == self.len:
            raise StopIteration
        else:
            ret_idx = self.idx
            ret_name = self.name
            ret_df = self.df.iloc[self.begin:self.end].reset_index(drop=True)
            self.next()
            return ret_idx, ret_name, ret_df


class DFContainer:
    def __init__(self, df):
        self.df = df.sort_values(by=['NAME']).reset_index(drop=True)

    def __iter__(self):
        return DFIterator(self.df)

    def column_list(self):
        return list(self.df)

    def get_via(self, name, idx):
        vias = self.df[(self.df['NAME'] == name) & (self.df['VIA_IDX'] == idx)]
        if vias.shape[0]:
            return vias.iloc[0]
        else:
            print('\nError: design {0} net {1} has no via {2}'.format(
                self.df.at[0, 'DESIGN'], name, idx))
            exit()

    def keys(self):
        return self.df[['NAME']].drop_duplicates().sort_values(by=['NAME']).reset_index(drop=True)


def process_net(net_info, layer, dtype, clean_img=False):
    if clean_img:
        global png_dict
        png_dict = {}

    for net in net_info:
        key = "{0}_{1}_{2}".format(net[0], net[1], net[2])
        prefix = "split-extract-ft-drv-grammar/{0}/{0}_M{1}/{2}_{3}".format(
            net[0], layer, net[1], net[2])
        png_dict[key] = np.zeros((99, 99, 3), dtype=dtype)
        png_dict[key][:, :, 0] = imageio.imread(
            "{0}_{1}.png".format(prefix, 1))
        png_dict[key][:, :, 1] = imageio.imread(
            "{0}_{1}.png".format(prefix, 2))
        png_dict[key][:, :, 2] = imageio.imread(
            "{0}_{1}.png".format(prefix, 4))


def get_data_sep(path, layer, dtype):
    net_indicator = ['DESIGN', 'PARENT', 'NAME', 'SINK_COUNT']
    via_indicator = ['DESIGN', 'NAME', 'VIA_IDX']

    drv_df = pd.read_csv(path + '.drv.csv', keep_default_na=False)
    snk_df = pd.read_csv(path + '.snk.csv', keep_default_na=False)
    snk_nets = snk_df[net_indicator].drop_duplicates(
    ).sort_values(by=['NAME']).reset_index(drop=True)

    sink_info = snk_df[via_indicator].values
    source_info = drv_df[via_indicator].values
    net_info = np.concatenate((sink_info, source_info))
    process_net(net_info, layer, dtype=dtype, clean_img=True)

    return DFContainer(drv_df), DFContainer(snk_df), snk_nets


def get_image_batch_sep(drv_df, snk_vias, _drv_nets, snk_name, drv_size, dtype):
    drv_nets = _drv_nets.copy()
    drv_nets['cost'] = sys.maxsize
    drv_nets['drv_idx'] = -1
    drv_nets['snk_idx'] = -1
    design = snk_vias.at[0, 'DESIGN']
    num_drv = drv_nets.shape[0]

    for drv_idx, drv_name, drv_vias in drv_df:
        if drv_name != drv_nets.at[drv_idx, 'NAME']:
            print('\n `drv_name` {0} does not match name {1} in `drv_nets`'.format(
                drv_name, drv_nets.at[drv_idx, 'NAME']))
            exit()

        for _, drv_via in drv_vias.iterrows():
            drv_via_x = drv_via['VIA_RELATIVE_X']
            drv_via_y = drv_via['VIA_RELATIVE_Y']
            drv_bnd = drv_via_x < 0.08 or drv_via_x > 0.92
            for _, snk_via in snk_vias.iterrows():
                costx = abs(drv_via['VIA_ABSOLUTE_X'] -
                            snk_via['VIA_ABSOLUTE_X'])
                if costx >= 4000:
                    continue

                costy = abs(drv_via['VIA_ABSOLUTE_Y'] -
                            snk_via['VIA_ABSOLUTE_Y'])
                if costy >= 4000:
                    continue

                cost = costx * 0x100000000 + costy
                if cost >= drv_nets.at[drv_idx, 'cost']:
                    continue

                prefer = False
                snk_via_x = snk_via['VIA_RELATIVE_X']
                snk_via_y = snk_via['VIA_RELATIVE_Y']
                if drv_via_x <= snk_via_x:
                    if drv_via_y <= snk_via_y:
                        prefer = prefer or drv_via['DIR_PP'] or snk_via['DIR_NN']

                    if drv_via_y >= snk_via_y:
                        prefer = prefer or drv_via['DIR_PN'] or snk_via['DIR_NP']

                if drv_via_x >= snk_via_x:
                    if drv_via_y <= snk_via_y:
                        prefer = prefer or drv_via['DIR_NP'] or snk_via['DIR_PN']

                    if drv_via_y >= snk_via_y:
                        prefer = prefer or drv_via['DIR_NN'] or snk_via['DIR_PP']

                snk_bnd = snk_via_x < 0.08 or snk_via_x > 0.92
                if num_drv <= drv_size or drv_bnd or snk_bnd or prefer:
                    drv_nets.at[drv_idx, 'cost'] = cost
                    drv_nets.at[drv_idx, 'drv_idx'] = drv_via['VIA_IDX']
                    drv_nets.at[drv_idx, 'snk_idx'] = snk_via['VIA_IDX']

    drv_nets = drv_nets[drv_nets['cost'] < sys.maxsize]
    if drv_size:
        drv_nets['select'] = 0
        drv_nets.sort_values(by=['cost'], inplace=True)
        drv_nets.reset_index(drop=True, inplace=True)
        for i in range(0, min(drv_size, drv_nets.shape[0])):
            drv_nets.at[i, 'select'] = 1

        drv_nets = drv_nets[drv_nets['select'] == 1]

    drv_nets.reset_index(drop=True, inplace=True)
    num_drv = drv_nets.shape[0]
    drv_imgs = np.zeros((num_drv, 99, 99, 3), dtype=dtype)
    snk_imgs = np.zeros((1, 99, 99, 3), dtype=dtype)
    snk_imgs[0, :, :, :] = png_dict["{0}_{1}_{2}".format(
        design, snk_name, snk_vias.at[0, 'VIA_IDX'])]
    data_col = drv_df.column_list()[8:]
    num_col = len(data_col)
    data = np.zeros((num_drv, 30), dtype=dtype)
    label = -1
    for index, drv_net in drv_nets.iterrows():
        drv_name = drv_net['NAME']
        drv_via_idx = drv_net['drv_idx']
        if drv_via_idx < 0:
            print('\nError: design {0} net {1} invalid via index {2}'.format(
                design, drv_name, drv_via_idx))
            exit()

        drv_via = drv_df.get_via(drv_name, drv_via_idx)
        drv_via_x = drv_via['VIA_ABSOLUTE_X']
        drv_via_y = drv_via['VIA_ABSOLUTE_Y']
        snk_idx = drv_net['snk_idx']
        snk_via = snk_vias[snk_vias['VIA_IDX'] == snk_idx].iloc[0]
        # SIGNED_ABSOLUTE_DIST_X
        data[index, 0] = (snk_via['VIA_ABSOLUTE_X'] - drv_via_x)
        # SIGNED_ABSOLUTE_DIST_Y
        data[index, 1] = (snk_via['VIA_ABSOLUTE_Y'] - drv_via_y)
        # SIGNED_RELATIVE_DIST_X
        data[index, 4] = snk_via['VIA_RELATIVE_X'] - drv_via['VIA_RELATIVE_X']
        # SIGNED_RELATIVE_DIST_Y
        data[index, 5] = snk_via['VIA_RELATIVE_Y'] - drv_via['VIA_RELATIVE_Y']
        # SNK_SINK_COUNT
        data[index, 8] = snk_via['SINK_COUNT']
        # DRV_SINK_COUNT
        data[index, 9] = drv_via['SINK_COUNT']
        # SNK_UP_PIN
        data[index, 10] = snk_via['UP_PIN']
        # DRV_UP_PIN
        data[index, 11] = drv_via['UP_PIN']
        data[index, 12:21] = snk_via[-9:]
        data[index, 21:30] = drv_via[-9:]
        drv_imgs[index, :, :, :] = png_dict["{0}_{1}_{2}".format(
            design, drv_name, drv_via_idx)]
        if drv_via['PARENT'] == snk_via['PARENT']:
            label = index

    # UNSIGNED
    data[:, 2:4] = abs(data[:, 0:2])
    data[:, 6:8] = abs(data[:, 4:6])

    return data, drv_imgs, snk_imgs, label


def get_data(path, design, layer, dtype, clean_img=False):
    image_indicator = ['design', 'SNK_NAME',
                       'DRV_NAME', 'SNK_VIA_IDX', 'DRV_VIA_IDX']
    snk_indicator = ['design', 'SNK_NAME', 'SNK_VIA_IDX']
    drv_indicator = ['design', 'DRV_NAME', 'DRV_VIA_IDX']

    df = pd.read_csv(path + '.sel.csv', keep_default_na=False)
    snsc = df[['SNK_SINK_COUNT']].values
    label = df[['LABEL']].values
    data = df.iloc[:, 4:-1].values.astype(dtype)
    sink = df.iloc[:, [0]].values.flatten()
    sink_name, sink_idx = np.unique(sink, return_inverse=True)
    df['design'] = design
    data[np.isnan(data)] = 0
    img_info = df[image_indicator].values
    sink_info = df[snk_indicator].drop_duplicates().values
    source_info = df[drv_indicator].drop_duplicates().values
    net_info = np.concatenate((sink_info, source_info))
    process_net(net_info, layer, dtype=dtype, clean_img=clean_img)

    return data, snsc, label, sink_name, sink_idx, img_info


def get_image_batch(image_infos, dtype):

    len_image = len(image_infos)
    image = np.zeros((1 + len_image, 99, 99, 3), dtype=dtype)
    image[0, :, :, :] = png_dict["{0}_{1}_{2}".format(
        image_infos[0, 0], image_infos[0, 1], image_infos[0, 3])]
    for i in range(0, len_image):
        image[1 + i, :, :, :] = png_dict["{0}_{1}_{2}".format(
            image_infos[i, 0], image_infos[i, 2], image_infos[i, 4])]

    return image
