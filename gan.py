# import tensorflow as tf 

# class GAN():
#     def __init__(self, args):
#         self.gen_dim = [1024, 512, 256, 128]
#         self.dis_dim = [64, 128, 256, 512]
#         self.image_size = args.image_size    #generated image size
#         self.z_dim = args.dim    #generator vector input size
#         self.batch_size = args.batch_size
#         self.istrain = args.istrain


#     def generator(self, data, name='gen_'):
#         with tf.variable_scope(name) as scope:
#             # 4 x 4
#             l1_fc = tf.layers.dense(data, 4*4*self.gen_dim[0])
#             l1_reshape = tf.reshape(l1_fc, [tf.shape(data)[0], 4, 4, self.gen_dim[0]])
# #             l1_reshape = tf.Print(l1_reshape, [l1_reshape.get_shape()])
#            # l1_batch_norm = tf.identity(l1_reshape)
#             l1_batch_norm = tf.layers.batch_normalization(l1_reshape,training=self.istrain)
#             l1_relu = tf.nn.relu(l1_batch_norm)

#             # 8 x 8
#             l2_deconv = tf.layers.conv2d_transpose(l1_relu, self.gen_dim[1], kernel_size = [3,3], 
#                         strides=(2,2), padding='same')
#             #l2_batch_norm = tf.identity(l2_deconv)
#             l2_batch_norm = tf.layers.batch_normalization(l2_deconv,training=self.istrain)
#             l2_relu = tf.nn.relu(l2_batch_norm)

#             # 16 x 16
#             l3_deconv = tf.layers.conv2d_transpose(l2_relu, self.gen_dim[2], kernel_size = [3,3], 
#                         strides=(2,2), padding='same')
#             #l3_batch_norm = tf.identity(l3_deconv)
#             l3_batch_norm = tf.layers.batch_normalization(l3_deconv,training=self.istrain)
#             l3_relu = tf.nn.relu(l3_batch_norm)

#             # 32 x 32
#             l4_deconv = tf.layers.conv2d_transpose(l3_relu, self.gen_dim[3], kernel_size = [3,3], 
#                         strides=(2,2), padding='same')
#             #l4_batch_norm = tf.identity(l4_deconv)
#             l4_batch_norm = tf.layers.batch_normalization(l4_deconv,training=self.istrain)
#             l4_relu = tf.nn.relu(l4_batch_norm)

#             # 64 x 64
#             output = tf.layers.conv2d_transpose(l4_relu, 3, kernel_size = [3,3], 
#                         strides=(2,2), padding='same')
#             return tf.nn.tanh(output)

    
#     def discriminator(self,data, name = 'dis_', reuse = False):
#         with tf.variable_scope(name) as scope:
#             if reuse==True:
#                 scope.reuse_variables()
#             # 64 -> 32
#             l1_conv = tf.layers.conv2d(data, self.dis_dim[0], kernel_size = [3,3], 
#                         strides = (1,1), padding = 'valid')
#             l1_leaky_relu = tf.nn.leaky_relu(l1_conv, alpha = 0.2)

#             # 32 -> 16
#             l2_conv = tf.layers.conv2d(l1_leaky_relu, self.dis_dim[1], kernel_size = [3,3], 
#                         strides = (1,1), padding = 'valid')
#             #l2_batch_norm = tf.identity(l2_conv)
#             l2_batch_norm = tf.layers.batch_normalization(l2_conv,training=self.istrain)
#             l2_relu = tf.nn.leaky_relu(l2_batch_norm, alpha = 0.2)

#             # 16 -> 8
#             l3_conv = tf.layers.conv2d(l2_relu, self.dis_dim[2], kernel_size = [3,3], 
#                         strides = (1,1), padding = 'valid')
#             #l3_batch_norm = tf.identity(l3_conv)
#             l3_batch_norm = tf.layers.batch_normalization(l3_conv,training=self.istrain)
#             l3_relu = tf.nn.leaky_relu(l3_batch_norm, alpha = 0.2)

#             # 8 -> 4
#             l4_conv = tf.layers.conv2d(l3_relu, self.dis_dim[3], kernel_size = [3,3], 
#                         strides = (1,1), padding = 'valid')
#            # l4_batch_norm = tf.identity(l4_conv)
#             l4_batch_norm = tf.layers.batch_normalization(l4_conv,training=self.istrain)
#             l4_relu = tf.nn.leaky_relu(l4_batch_norm, alpha = 0.2)

#             l4_flatten = tf.contrib.layers.flatten(l4_relu)

#             logits = tf.layers.dense(l4_flatten, 1)

#             return logits       


#     def build_model(self):
#         z = tf.placeholder(tf.float32, shape = [None, self.z_dim], name = 'z')
#         d = tf.placeholder(tf.float32, shape= [None, self.image_size, self.image_size, 3], name = 'd')

#         gen_output = self.generator(z)
#         gen_output_vis = gen_output*255
#         viz = tf.cast(gen_output_vis, tf.uint8, name='gen_im')
#         tf.summary.image('gen_im', viz, max_outputs=20)
#         logits_fake = self.discriminator(gen_output)     
#         logits_real = self.discriminator(d, reuse=True)

#         #generator loss
        
#         #G_loss = -tf.reduce_mean(tf.log(logits_fake))
#         G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(logits_fake)))
#         tf.summary.scalar('G_loss', G_loss)

#         #discriminator loss
#         #D_loss = -tf.reduce_mean(tf.log(logits_real) + tf.log(1-logits_fake))
#         D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = tf.ones_like(logits_real)))
#         D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.zeros_like(logits_fake)))
#         D_loss = D_loss_real + D_loss_fake
    
#         tf.summary.scalar('D_loss', D_loss)
        
#         return z,d,G_loss, D_loss

import tensorflow as tf 

class GAN():
    def __init__(self, args):
        self.gen_dim = [1024, 512, 256, 128]
        self.dis_dim = [64, 128, 256, 512]
        self.image_size = args.image_size    #generated image size
        self.z_dim = args.dim    #generator vector input size
        self.batch_size = args.batch_size
        self.istrain = args.istrain


    def generator(self, data, name='gen_'):
        with tf.variable_scope(name) as scope:
            # 4 x 4
            l1_fc = tf.layers.dense(data, 4*4*self.gen_dim[0])
            l1_reshape = tf.reshape(l1_fc, [tf.shape(data)[0], 4, 4, self.gen_dim[0]])
#             l1_reshape = tf.Print(l1_reshape, [l1_reshape.get_shape()])
#             l1_batch_norm = tf.identity(l1_reshape)
            l1_batch_norm = tf.layers.batch_normalization(l1_reshape,training=self.istrain)
            l1_relu = tf.nn.leaky_relu(l1_batch_norm, alpha = 0)

            # 8 x 8
            l2_deconv = tf.layers.conv2d_transpose(l1_relu, self.gen_dim[1], kernel_size = [3,3], 
                        strides=(2,2), padding='same')
#             l2_batch_norm = tf.identity(l2_deconv)
            l2_batch_norm = tf.layers.batch_normalization(l2_deconv,training=self.istrain)
            l2_relu = tf.nn.leaky_relu(l2_batch_norm, alpha = 0)

            # 16 x 16
            l3_deconv = tf.layers.conv2d_transpose(l2_relu, self.gen_dim[2], kernel_size = [3,3], 
                        strides=(2,2), padding='same')
#             l3_batch_norm = tf.identity(l3_deconv)
            l3_batch_norm = tf.layers.batch_normalization(l3_deconv,training=self.istrain)
            l3_relu = tf.nn.leaky_relu(l3_batch_norm, alpha = 0)

            # 32 x 32
            l4_deconv = tf.layers.conv2d_transpose(l3_relu, self.gen_dim[3], kernel_size = [3,3], 
                        strides=(2,2), padding='same')
#             l4_batch_norm = tf.identity(l4_deconv)
            l4_batch_norm = tf.layers.batch_normalization(l4_deconv,training=self.istrain)
            l4_relu = tf.nn.leaky_relu(l4_batch_norm, alpha = 0)

            # 64 x 64
            output = tf.layers.conv2d_transpose(l4_relu, 3, kernel_size = [3,3], 
                        strides=(2,2), padding='same')

            # gen_output
            output = tf.nn.tanh(output, name = "tanh")
            return output

    
    def discriminator(self,data, name = 'dis_', reuse = False):
        with tf.variable_scope(name) as scope:
            if reuse==True:
                scope.reuse_variables()
            # 64 -> 32
            l1_conv = tf.layers.conv2d(data, self.dis_dim[0], kernel_size = [3,3], 
                        strides = (1,1), padding = 'valid')
#             l1_batch_norm = tf.identity(l1_conv)
            l1_batch_norm = tf.layers.batch_normalization(l1_conv,training=self.istrain)
            l1_leaky_relu = tf.nn.leaky_relu(l1_batch_norm, alpha = 0.2)

            # 32 -> 16
            l2_conv = tf.layers.conv2d(l1_leaky_relu, self.dis_dim[1], kernel_size = [3,3], 
                        strides = (1,1), padding = 'valid')
#             l2_batch_norm = tf.identity(l2_conv)
            l2_batch_norm = tf.layers.batch_normalization(l2_conv,training=self.istrain)
            l2_relu = tf.nn.leaky_relu(l2_batch_norm, alpha = 0.2)

            # 16 -> 8
            l3_conv = tf.layers.conv2d(l2_relu, self.dis_dim[2], kernel_size = [3,3], 
                        strides = (1,1), padding = 'valid')
#             l3_batch_norm = tf.identity(l3_conv)
            l3_batch_norm = tf.layers.batch_normalization(l3_conv,training=self.istrain)
            l3_relu = tf.nn.leaky_relu(l3_batch_norm, alpha = 0.2)

            # 8 -> 4
            l4_conv = tf.layers.conv2d(l3_relu, self.dis_dim[3], kernel_size = [3,3], 
                        strides = (1,1), padding = 'valid')
#             l4_batch_norm = tf.identity(l4_conv)
            l4_batch_norm = tf.layers.batch_normalization(l4_conv,training=self.istrain)
            l4_relu = tf.nn.leaky_relu(l4_batch_norm, alpha = 0.2)

            l4_flatten = tf.contrib.layers.flatten(l4_relu)

            logits = tf.layers.dense(l4_flatten, 1)

            return logits       


    def build_model(self):
        self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim], name = 'z')
        d = tf.placeholder(tf.float32, shape= [None, self.image_size, self.image_size, 3], name = 'd')

        self.gen_output = self.generator(self.z)
        gen_output_vis = (self.gen_output+1)/2*255
        viz = tf.cast(gen_output_vis, tf.uint8, name='gen_im')
        tf.summary.image('gen_im', viz, max_outputs=20)
        
        logits_fake = self.discriminator(self.gen_output)  
        d_viz = (d+1)/2*255
        d_viz = tf.cast(d_viz,  tf.uint8, name='real_im')
        tf.summary.image('real_im', d_viz, max_outputs=20)
        logits_real = self.discriminator(d, reuse=True)

        #generator loss
        
        #G_loss = -tf.reduce_mean(tf.log(logits_fake))
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(logits_fake)), name = "reduced_mean")
#         G_loss_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(logits_fake), name = 'cross_entropy')
        tf.summary.scalar('G_loss', self.G_loss)

        #discriminator loss
        #D_loss = -tf.reduce_mean(tf.log(logits_real) + tf.log(1-logits_fake))
#         D_acc = 
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_real, labels = tf.ones_like(logits_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.zeros_like(logits_fake)))
        D_loss = D_loss_real + D_loss_fake
    
        tf.summary.scalar('D_loss', D_loss)
        
        return self.z,d, D_loss, self.G_loss



        


       


        
     

      






      
        






        


       


        
     

      



