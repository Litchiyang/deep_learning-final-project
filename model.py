import tensorflow as tf
import numpy as np
import gan
import poissonblending as poissonblending
from scipy.signal import convolve2d
import config
import train


class ModelInpaint():
    def __init__(self, modelfilename, conf, model_name='dcgan',
                gen_input='z:0', gen_output='gen_/tanh:0', gen_loss='reduced_mean:0',
                z_dim=100, batch_size=64):

        self.conf = conf
        print(config)
        print ("lala")
        self.args = train.configure()
        print("aha")

        self.batch_size = batch_size
        self.z_dim = z_dim

        # self.gi = self.graph.get_tensor_by_name(model_name+'/'+gen_input)
        # self.go = self.graph.get_tensor_by_name(model_name+'/'+gen_output)
        # self.gl = self.graph.get_tensor_by_name(model_name+'/'+gen_loss)

        self.gcgan = gan.GAN(args)

        self.gi = gcgan.z
        self.go = gcgan.gen_output
        self.gl = gcgan.G_loss

        self.image_shape = self.go.shape[1:].as_list()

        self.lamb = config.lambda_p

        self.sess = tf.Session(graph=self.graph)

        self.z = np.random.randn(self.batch_size, self.z_dim)
        

    def sample(self, z=None):
        """GAN sampler. Useful for checking if the GAN was loaded correctly"""
        if z is None:
            z = self.z
        sample_out = self.sess.run(self.go, feed_dict={self.gi: z})
        return sample_out

    def preprocess(self, images, imask, useWeightedMask = True, nsize=7):

        images = np.array(images) / 127.5-1
        if useWeightedMask:
            ker = np.ones((nsize,nsize), dtype=np.float32)
            ker = ker/np.sum(ker)
            mask = imask * convolve2d(imask, ker, mode='same', boundary='symm')
        else:
            mask = imask
 
        mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)

        bin_mask = np.array(imask, dtype=np.float32)
        bin_mask[bin_mask>0] = 1.0
        bin_mask[bin_mask<=0] = 0

        self.bin_mask = np.repeat(bin_mask[:,:,np.newaxis], 3, axis=2)

        self.masks_data = np.repeat(mask[np.newaxis, :, :, :],
                                    self.batch_size,
                                    axis=0)

        if len(images.shape) is 3:
            self.images_data = np.repeat(images[np.newaxis, :, :, :],
                                         self.batch_size,
                                         axis=0)
        elif len(images.shape) is 4:
            num_images = images.shape[0]
            self.images_data = np.repeat(images[np.newaxis, 0, :, :, :],
                                         self.batch_size,
                                         axis=0)
            ncpy = min(num_images, self.batch_size)
            self.images_data[:ncpy, :, :, :] = images[:ncpy, :, :, :].copy()


    def postprocess(self, g_out):

        images_out = (np.array(g_out) + 1) / 2.0
        images_in = (np.array(self.images_data) + 1) / 2.0

        for i in range(len(g_out)):
            images_out[i] = poissonblending.blend(images_in[i], images_out[i], 1-self.bin_mask)

        else:
            images_out = np.multiply(images_out, 1-self.masks_data) + np.multiply(images_in, self.masks_data)

        return images_out


    def build_inpaint_graph(self):
        with self.graph.as_default():
            self.masks = tf.placeholder(tf.float32, [None, 64, 64, 3], name='mask')
            self.images = tf.placeholder(tf.float32, [None, 64, 64, 3], name='image')

            context_loss = tf.abs(tf.multiply(self.masks, self.go) - tf.multiply(self.masks, self.images))
            self.context_loss = tf.reduce_sum(tf.contrib.layers.flatten(context_loss), 1)

            self.prior_loss = self.gl
            self.loss = self.context_loss + self.lamb * self.prior_loss
            self.grad = tf.gradients(self.loss, self.gi)


    def inpaint(self, image, mask):
        v = 0
        self.build_inpaint_graph()
        self.preprocess(image, mask)

        for i in range(self.config.nIter):
            out_vars = [self.loss, self.grad, self.go]
            in_dict = {self.masks: self.masks_data,
                       self.gi: self.z,
                       self.images: self.images_data}

            loss, grad, imout = self.sess.run(out_vars, feed_dict=in_dict)

            v_prev = np.copy(v)
            v = self.config.momentum*v - self.config.lr*grad[0]
            self.z += (-self.config.momentum * v_prev + (1 + self.config.momentum) * v)
            self.z = np.clip(self.z, -1, 1)

            print('Iteration {}: {}'.format(i, np.mean(loss)))

        blend_output = self.postprocess(imout)

        return blend_output, imout
    




