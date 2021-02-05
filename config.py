import numpy as np


class Config:
    epoch = 400
    show_epoch = 10
    data_path = './data'
    dtype32 = np.float32
    dtype64 = np.float64


class ConfigFanout(Config):
    learning_rate = 0.0005
    lr_decay_factor = 0.4

    def __init__(self, design, layer, iteration):
        self.design = design
        self.layer = layer
        self.iteration = iteration
        self.start = self.iteration * 100 + 1

        self.model_path = './checkpoints-rand-' + self.design

        self.train_list = []
        for i in [self.design]:  # 'b14', 'b15', 'b17', 'b20', 'b21', 'b22'
            for j in range(self.start, self.start + 40):
                self.train_list.append('{0}_C_{1}'.format(i, j))

        self.train_layer = [self.layer] * len(self.train_list)

        self.cv_list = []
        for i in [self.design]:  # 'b14', 'b15', 'b17', 'b20', 'b21', 'b22'
            for j in range(self.start + 40, self.start + 50):
                self.cv_list.append('{0}_C_{1}'.format(i, j))

        self.cv_layer = [self.layer] * len(self.cv_list)

        self.attack_list = []
        for i in [self.design]:  # 'b14', 'b15', 'b17', 'b20', 'b21', 'b22'
            for j in range(self.start + 50, self.start + 96):
                self.attack_list.append('{0}_C_{1}'.format(i, j))

        self.attack_layer = [self.layer] * len(self.attack_list)

        self.test_list = self.attack_list + self.train_list + self.cv_list
        self.test_layer = [self.layer] * len(self.test_list)


class ConfigLPDC(Config):
    learning_rate = 0.0003
    lr_decay_factor = 0.7
    model_path = "./checkpoints-lpdc"
    train_list = ['LPDC_iter2', 'LPDC_iter3', 'LPDC_iter4', 'LPDC_iter5',
                  'LPDC_iter6', 'LPDC_iter7', 'LPDC_iter8', 'LPDC_iter9']
    train_layer = [8] * len(train_list)
    cv_list = ['LPDC_iter1', 'LPDC_iter10']
    cv_layer = [8] * len(cv_list)
    attack_list = ['LPDC_iter0']
    attack_layer = [8] * len(attack_list)
    test_list = attack_list + train_list + cv_list
    test_layer = [8] * len(test_list)


class ConfigRand(Config):
    learning_rate = 0.0005
    lr_decay_factor = 0.4

    def __init__(self, design, layer, iteration):
        self.design = design
        self.layer = layer
        self.iteration = iteration
        self.start = self.iteration * 100 + 1

        self.model_path = './checkpoints-rand-' + self.design

        self.train_list = []
        for i in ['b14', 'b15', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22']:
            if i == self.design:
                continue

            for j in range(self.start, self.start + 10):
                self.train_list.append('{0}_C_{1}'.format(i, j))

        self.train_layer = [self.layer] * len(self.train_list)

        self.cv_list = []
        for i in ['b14', 'b15', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22']:
            if i == self.design:
                continue

            for j in range(self.start + 40, self.start + 42):
                self.cv_list.append('{0}_C_{1}'.format(i, j))

        self.cv_layer = [self.layer] * len(self.cv_list)

        self.attack_list = []
        for i in [self.design]:  # 'b14', 'b15', 'b17', 'b20', 'b21', 'b22'
            for j in range(self.start, self.start + 96):
                self.attack_list.append('{0}_C_{1}'.format(i, j))

        self.attack_layer = [self.layer] * len(self.attack_list)

        self.test_list = self.attack_list + self.train_list + self.cv_list
        self.test_layer = [self.layer] * len(self.test_list)


class ConfigTCAD20(Config):
    learning_rate = 0.0006
    lr_decay_factor = 0.7
    model_path = "./checkpoints-tcad20"
    train_list = ['c6288-asap7', 'b14-asap7', 'b17-a64', 'b18-a64']
    train_layer = [3] * len(train_list)
    cv_list = ['c7552-a64', 'b15-asap7']
    cv_layer = [3] * len(cv_list)
    attack_list = ['c432_C-a64', 'c880_C16-a64', 'c1355_C-a64', 'c1908_C-a64',
                   'c2670_C-a64', 'c3540_C-a64', 'c5315_C-a64', 'c6288_C-a64',
                   'b07_C-a64', 'b11_C-a64', 'b13_C-a64', 'b14_C-a64', 'b15_C-a64', 'b17_C-a64']
    attack_layer = [3] * len(attack_list)
    test_list = attack_list + train_list + cv_list
    test_list += ['c432-asap7', 'c432_C-asap7', 'c432_C13-asap7', 'c880-asap7',
                  'c880_C-asap7', 'c880_C17-a64', 'c1355-asap7',
                  'c1355_C-asap7', 'c1908-asap7', 'c1908_C-asap7',
                  'c2670-asap7', 'c2670_C-asap7', 'c3540-asap7',
                  'c3540_C-asap7', 'c5315-asap7', 'c5315_C-asap7',
                  'c6288_C-asap7', 'c7552-asap7', 'b07-asap7', 'b07_C-asap7', 'b11-asap7', 'b11_C-asap7', 'b13-asap7', 'b13_C-asap7', 'b14_C-asap7',
                  'b15_C-asap7', 'b17-asap7', 'b17_C-asap7', 'b18-asap7']
    test_layer = [3] * len(test_list)


class ConfigTVLSI19(Config):
    learning_rate = 0.0003
    lr_decay_factor = 0.7
    model_path = "./checkpoints"
    train_list = ['apex2', 'b20', 'b21', 'des', 'c499', 'dalu', 'i4', 'i8',
                  'i9', 'k2', 'seq']
    train_layer = [3] * len(train_list)
    cv_list = ['ex1010']
    cv_layer = [3] * len(cv_list)
    attack_list = ['c432', 'c1908', 'c1355', 'c880', 'b7', 'b13', 'b11',
                   'c6288', 'c2670', 'c3540', 'c7552', 'c5315', 'b14', 'b15_1',
                   'b17_1', 'b18']
    attack_layer = [3] * len(attack_list)
    test_list = attack_list + train_list + cv_list
    test_list += ['apex4', 'b22', 'ex5', 'i7']
    test_layer = [3] * len(test_list)


class ConfigTVLSI19M3(Config):
    learning_rate = 0.0005
    lr_decay_factor = 0.7
    model_path = "./checkpoints-tvlsi19-m3"
    train_list = ['apex2', 'b20', 'b21',  'c499',
                  'dalu', 'des', 'i4', 'i8', 'i9', 'k2', 'seq']
    train_layer = [3] * len(train_list)
    cv_list = ['ex1010']
    cv_layer = [3] * len(cv_list)
    attack_list = ['c432', 'c1908', 'c1355', 'c880', 'b7', 'b13', 'b11',
                   'c6288', 'c2670', 'c3540', 'c7552', 'c5315', 'b14', 'b15_1',
                   'b17_1', 'b18']
    attack_layer = [3] * len(attack_list)
    test_list = attack_list + train_list + cv_list
    test_list += ['apex4', 'b22', 'ex5', 'i7']
    test_layer = [3] * len(test_list)


class ConfigTVLSI19M5(Config):
    learning_rate = 0.0008
    lr_decay_factor = 0.7
    model_path = "./checkpoints-tvlsi19-m5"
    train_list = ['apex4', 'b19_1', 'c499', 'dalu', 'des', 'ex1010',
                  'ex5', 'i4', 'i7', 'i8', 'i9', 'k2', 'seq']  # 'apex2',
    train_layer = [5] * len(train_list)
    cv_list = ['b14']
    cv_layer = [5] * len(cv_list)
    attack_list = ['c432', 'c1908', 'c1355', 'c880', 'b7', 'b13', 'b11', 'c2670',
                   'c3540', 'c7552', 'c5315', 'b14', 'b15_1', 'b17_1', 'b18']  # 'c6288',
    attack_layer = [5] * len(attack_list)
    test_list = attack_list + train_list + cv_list
    test_list += ['b19', 'b20', 'b21', 'b22']
    test_layer = [5] * len(test_list)


class ConfigTVLSI19M5Pixel(Config):
    learning_rate = 0.00005
    lr_decay_factor = 0.7
    model_path = "./checkpoints-tvlsi19-m5-pixel"
    train_list = ['b14_C_0', 'b15_C_0', 'b17_C_0',
                  'b20_C_0', 'b21_C_0', 'b22_C_0']
    train_layer = [5] * len(train_list)
    cv_list = ['b14_C_0']
    cv_layer = [5] * len(cv_list)
    attack_list = []
    attack_layer = [5] * len(attack_list)
    test_list = attack_list + train_list + cv_list
    test_layer = [5] * len(test_list)


class ConfigTVLSI19M6(Config):
    learning_rate = 0.001
    lr_decay_factor = 0.7
    model_path = "./checkpoints-tvlsi19-m6"
    train_list = ['apex4', 'b18', 'b18_1', 'b19', 'b19_1', 'c499', 'dalu',
                  'des', 'ex1010', 'i4', 'i7', 'i8', 'i9', 'k2', 'seq']  # 'apex2', 'ex5',
    train_layer = [6] * len(train_list)
    cv_list = ['b22_C_0']
    cv_layer = [6] * len(cv_list)
    attack_list = ['b14_C_0', 'b15_C_0', 'b17_C_0',
                   'b20_C_0', 'b21_C_0', 'b22_C_0']
    attack_layer = [6] * len(attack_list)
    test_list = attack_list + train_list + cv_list
    test_list += ['b19', 'b20', 'b21', 'b22']
    test_layer = [6] * len(test_list)
