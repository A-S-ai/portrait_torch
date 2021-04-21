class Config(object):

    def __init__(self, server=False):

        self.num_channels = 3
        self.num_classes = 1

        self.epoch = 20
        self.batch_size = 4
        self.lr = 0.0001
        self.val_rate = 0.1
        self.num_workers = 2
        self.pin_m = False

        self.data_root = './data'
        self.model_root = './result'
        self.log_root = './log'

        if server:
            self.num_workers = 4
            self.batch_size = 12
            self.pin_m = True

    def roots(self):
        return self.data_root, self.model_root, self.log_root
