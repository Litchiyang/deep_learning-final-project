import tensorflow as tf
import numpy as np

# possion blending is from external resource: https://github.com/parosky/poissonblending 
import poissonblending as poissonblending
from scipy.signal import convolve2d


class inpaint_model():
    def __init__(self, modelfilename, z_dimension=100, batch_size=9, lamb=0.01, iteration=5, momentum=0.9, learning_rate=0.01):

        self.batch_size = batch_size
        self.lamb = lamb
        self.iteration = iteration
        self.momentum = momentum
        self.lr = learning_rate

        self.graph = inpaint_model.build_graph(modelfilename)

        self.g_input = self.graph.get_tensor_by_name('dcgan/z:0')
        self.g_output = self.graph.get_tensor_by_name('dcgan/Tanh:0')
        self.g_loss = self.graph.get_tensor_by_name('dcgan/Mean_2:0')

        # initialize vector z
        self.z = np.random.randn(self.batch_size, z_dimension)


    def build_graph(filename):

        # load graph definition from pb file
        with tf.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # build graph from graph definition
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='dcgan')

        return graph


    def preprocess(self, images, in_mask, win_size=7):

        # Compute the weighted masks W and repeat it to match the picture size and batch size
        N = np.ones((win_size, win_size), dtype=np.float32)
        mask = in_mask * convolve2d(in_mask, N/np.sum(N), mode='same', boundary='symm')
 
        mask = np.repeat(mask[:,:, None], 3, axis=2)

        # Add binary mask
        binary_mask = np.array(in_mask, dtype=np.float32)
        binary_mask[binary_mask > 0] = 1.0
        binary_mask[binary_mask <= 0] = 0

        self.binary_mask = np.repeat(binary_mask[:,:, None], 3, axis=2)

        self.batch_data = np.repeat(mask[None, :, :, :], self.batch_size, axis=0)

        # Prepare the batch images
        self.input_images = np.repeat(images[None, 0, :, :, :], self.batch_size, axis=0)
        self.input_images[:min(images.shape[0], self.batch_size), :, :, :] = images[:min(images.shape[0], self.batch_size), :, :, :].copy()
        self.vel = 0


    def blend(self, g_out):

        # Conduct Poission Blending to the output
        out_im = (np.array(g_out) + 1)/2
        in_im = (np.array(self.input_images)+1)/2

        for i in range(len(g_out)):
            out_im[i] = poissonblending.blend(in_im[i], out_im[i], 1-self.binary_mask)

        return out_im


    def build_inpaint_graph(self):

        with self.graph.as_default():

            self.mask_ph = tf.placeholder(tf.float32, [None, 64, 64, 3], name='mask')
            self.image_ph = tf.placeholder(tf.float32, [None, 64, 64, 3], name='image')

            # calculate loss of input
            context_loss = tf.abs(tf.multiply(self.mask_ph, self.g_output) - tf.multiply(self.mask_ph, self.image_ph))
            self.context_loss = tf.reduce_sum(tf.contrib.layers.flatten(context_loss), 1)

            self.prior_loss = self.g_loss
            self.loss = self.context_loss + self.lamb * self.prior_loss
            self.grad = tf.gradients(self.loss, self.g_input)


    def inpaint(self, image, mask):
        self.build_inpaint_graph()
        self.preprocess(image, mask)

        with tf.Session(graph=self.graph) as sess:   

            for i in range(self.iteration):

                loss, grad, self.out = sess.run([self.loss, self.grad, self.g_output], 
                                                    feed_dict  ={self.mask_ph: self.batch_data,
                                                                self.g_input: self.z,
                                                                self.image_ph: self.input_images})

                v_ = np.copy(self.vel)
                self.vel = self.momentum * self.vel - self.lr * grad[0]
                self.z += (-self.momentum * v_ + (1 + self.momentum) * self.vel)
                self.z = np.clip(self.z, -1, 1)

                print('iteration: {}, loss: {}'.format(i, np.mean(loss)))

        blend_output = self.blend(self.out)

        return blend_output, self.out
