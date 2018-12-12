import os
import numpy as np
import glob
import utils as utils


class CelebA(object):
    def __init__(self, dataset):
        self.name = dataset
        self.train_pics
        self.val_pics
        self.load()
        
        
    def load(self):
        print('Loading {}...'.format(self.name))
        self.train_pics = [pic for pic in os.listdir('./celebA/train') if os.path.isfile(pic)]    #under same directory as current file
        self.val_pics = [pic for pic in os.listdir('./celebA/val') if os.path.isfile(pic)]

        
    def train_next_batch(self, batch_size):
        pics = self.train_pics
        random_pics = np.random.choice(pics, batch_size)
        images = [utils.load_data(path, input_height=108, input_width=108) for path in random_pics]

        return np.asarray(images)

    
    def val_next_batch(self, batch_size):
        pics = self.val_pics
        random_pics = np.random.choice(pics, batch_size)
        images = [utils.load_data(path, input_height=108, input_width=108) for path in random_pics]

        return np.asarray(images)
