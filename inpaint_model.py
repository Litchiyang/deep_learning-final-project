"""1.需要DCGAN的相关函数（用||  ||标出），以及如何reload训练好的模型（对应原文loadpb函数，这里没有写，因为不确定适用不适用我们自己训练的模型）"""
"""2.Tensorflow相关部分，如self.sess部分没有完成"""
"""3.标有suppose部分为简化的地方，关系到mask, image输入的时候格式大小问题，如不使用可直接采用原文的处理方式"""
"""4.Poission Blending函数的引入"""




import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d

from ? import poissionblending                                              ## poission blending 
from ? import DCGAN                                                    #DCGAN: how to load?
 

class Inpaint():
    
    def __init__(self, batch_size = 64, D_z = 100, lam = 0.003, iter_n = 1500):
        
        self.batch_size = batch_size
        self.D_z = D_z
        self.lam = lam
        self.iter_n = iter_n
        self.z = np.random.randn(batch_size, D_z)
        
        ||self.sess = tf.Session(graph=self.graph)||                               ### Tensorflow 
        
    
        
        
        
    
    def pre(self, mask, images, win_size = 7):
        """1.Compute weighted mask W; 2.将mask、image根据batch_size进行调整 对应preprocess"""
        
        batch_size = self.batch_size
        
        #Compute the weighted masks W and repeat it to match the picture size and batch size.
        N = np.ones((win_size, win_size), dtype = np.float32 )
        wmask = mask * convolve2d(1 - mask, N/np.sum(N), mode = 'same', boundary = 'symm')        ##1 - mask or mask?
        wmask = wmask[:,:,None]
        wmasks = np.repeat (wmask, 3, axis = 2)
        self.batch_wmasks = np.repeat(wmasks[None, :, :, :], batch_size, axis = 0)
        
        self.ori_mask = np.repeat (mask[:,:,None], 3, axis = 2)                          ##Suppose all mask is binary
        
        #Prepare the batch images 
        images = images *2 / (256 - 1) -1
        self.batch_images = np.repeat(images[None, :, :, :], batch_size, axis=0)               ##Suppose images.dim = 3
        
        
    def post(self, g_output):
        """对最终结果进行poission blending， 对应postprocess"""
        
        batch_images = self.batch_images
        ori_mask = self.ori_mask
        
        #Conduct Poission Blending to the output
        out_img = (np.array(g_output) + 1) / 2
        in_img = (np.array(batch_images) + 1) / 2
        n = len(g_output)
        for i in range (n):
            out_img[i] = poissionblending(in_img[i], out_img[i], ori_mask)
            
        return out_img
        
    
    def loss_func(self):
        """构建loss function,对应 build_inpaint_graph"""
        
        # Set the loss function which is mentioned in the paper
        
        #with self.graph.as_default():                                       ###self.graph -- load model?
            
        self.mask = tf.placeholder(tf.float32, [None, 64,64,3] , name = 'mask' )           ##suppose image_shape is (64,64,3) ?
        self.image = tf.placeholder(tf.float32, [None, 64,64,3] , name = 'image' )
        
         
        context_loss = tf.abs(tf.multiply(self.mask, self.image) - tf.multiply(self.mask, ||self.DCGAN.gout||))   # DCGAN
        self.context_loss = tf.reduce_sum(tf.contrib.layers.flatten(context_loss), 1)
        
        self.prior_loss = self.lam * ||self.DCGAN.g_loss||                                         #DCGAN
        
        self.loss = self.context_loss + self.prior_loss
        
        self.grad = tf.gradients(self.loss, ||self.DCGAN.z||)                                       #DCGAN
        
   
    def inpaint(self, image, mask):
        """Do the backprop(optimization), 对应backprop_to_input及inpaint"""
        
        self.learning_rate = ?
        self.momentum = ?
        self.vel = 0
        
        self.pre(mask, images, win_size = 7)
        self.loss_func()
        
        iter_n = self.iter_n
        
        #Do the optimization
        for i in range (iter_n):
            
            loss, grad, output = ||self.sess.run|| ([self.loss, self.grad, ||self.DCGAN.gout||], feed.dict = {self.mask : self.batch_wmasks, ||self.DCGAN.in|| :self.z, self.image : self.batch_images} )
            
            vel_p = np.copy(self.vel)
            self.vel = self.momentum * self.vel - self.learning_rate * grad[0]
            self.z = self.z + (1 + self.momentum) * self.vel - self.momentum * vel_p
            
            self.z = np.clip(self.z , -1, 1)
            
        blend_output = self.post(output)  
        
        return blend_output, putput
            
            
