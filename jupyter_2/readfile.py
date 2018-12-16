
import os
import numpy as np
import glob
import matplotlib.image as img
import scipy.misc
import utils



# class CelebA(object):
#     def __init__(self, dataset_path):
#         self.name = dataset_path
#         self.dataset_path = dataset_path
#         self.train_path = os.path.join(dataset_path, 'img_align_celeba')
   
#         self.val_path = os.path.join(dataset_path, 'img_align_celeba')
        
#         # self.train_pics
#         # self.val_pics
#         self.load()
        
        
#     def load(self):
#         print('Loading {}...'.format(self.name))
      
#         self.train_pics = [os.path.join(self.train_path, pic) for pic in os.listdir(self.train_path) if ".jpg" in pic]    #under same directory as current file   
#         self.val_pics = [os.path.join(self.val_path, pic) for pic in os.listdir(self.val_path) if ".jpg" in pic]

        
#     def train_next_batch(self, batch_size, image_size):
#         pics = self.train_pics
#         random_pics = np.random.choice(pics, batch_size)
#         images = [utils.load_data(path, 64, 64) for path in random_pics]

#         return np.asarray(images)

    
#     def val_next_batch(self, batch_size):
#         pics = self.val_pics
#         random_pics = np.random.choice(pics, batch_size)
#         images = [utils.load_data(path, 64, 64) for path in random_pics]

#         return np.asarray(images)
    
#     def load_data(path):
#         height_resize=64
#         width_resize=64
#         image = scipy.misc.imread(path, mode="RGB").astype(np.float)
#         image = scipy.misc.imresize(image, (height_resize, width_resize)) 

#         return image


class CelebA(object):
    def __init__(self, dataset_path):
        self.name = dataset_path
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, 'img_align_celeba')
   
        self.val_path = os.path.join(dataset_path, 'img_align_celeba')
        
        # self.train_pics
        # self.val_pics
        self.load()
        
        
    def load(self):
        print('Loading {}...'.format(self.name))
      
        self.train_pics = [os.path.join(self.train_path, pic) for pic in os.listdir(self.train_path) if ".jpg" in pic]    #under same directory as current file   
        self.val_pics = [os.path.join(self.val_path, pic) for pic in os.listdir(self.val_path) if ".jpg" in pic]

        
    def train_next_batch(self, batch_size, image_size):
        pics = self.train_pics
        random_pics = np.random.choice(pics, batch_size)
        images = [utils.load_data(path, input_height=image_size, input_width=image_size) for path in random_pics]
        return np.asarray(images)

    
    def val_next_batch(self, batch_size):
        pics = self.val_pics
        random_pics = np.random.choice(pics, batch_size)
        images = [utils.load_data(path, input_height=image_size, input_width=image_size) for path in random_pics]

        return np.asarray(images)