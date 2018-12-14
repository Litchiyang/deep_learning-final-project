import os
import numpy as np
import glob
import matplotlib.image as img



class CelebA(object):
    def __init__(self, dataset_path):
        self.name = dataset_path
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, './celebA/train')
        self.val_path = os.path.join(dataset_path, './celebA/val')
        self.load()
        
        
    def load(self):
        print('Loading {}...'.format(self.name))
        print('under directory'.format(os.getcwd()))
        
        self.train_pics = [os.path.basename(pic) for pic in os.listdir(self.train_path) if os.path.isfile(pic)]
        self.val_pics = [os.path.basename(pic) for pic in os.listdir(self.val_path) if os.path.isfile(pic)]
        
        print('{} dataset finished loading'.format(self.name))

        
    def train_next_batch(self, batch_size):
        pics = self.train_pics
        random_pics = np.random.choice(pics, batch_size)
        images = [load_data(path) for path in random_pics]

        return np.asarray(images)

    
    def val_next_batch(self, batch_size):
        pics = self.val_pics
        random_pics = np.random.choice(pics, batch_size)
        images = [load_data(path) for path in random_pics]

        return np.asarray(images)
    
    

def load_data(path, height_resize=64, width_resize=64)

    image = cv2.resize(img.imread(path), (height, width_resize)) 
    
    return image