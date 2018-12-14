import tensorflow as tf
import numpy as np
import os
import config
import gan
import readfile

def train(args):
    #read data
    data_cele = readfile.CelebA(args.dataset_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)   #create a session

    #create an GAN object
    gan_model = gan.GAN(args) 

    #build a graph and get the loss
    z,d,D_loss, G_loss = gan_model.build_model()

    #optimizer 
    theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis_')
    d_optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate)\
                .minimize(D_loss, var_list = theta_D)

    theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen_')
    g_optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate)\
                .minimize(G_loss, var_list=theta_G)
    
    #initialize
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
            
    for i in range(args.iteration):
        _, D_loss_curr = session.run([d_optimizer, D_loss], feed_dict={z: sample_z(args.batch_size, args.dim), d: np.ones((args.batch_size, 64,64,3))})
        _, G_loss_curr = session.run([g_optimizer, G_loss], feed_dict={z: sample_z(args.batch_size, args.dim), d: np.ones((args.batch_size, 64,64,3))})    


def sample_z(batch_size, dim):
    return np.random.uniform(-1., 1., size=[batch_size, dim])



def configure():
    script_name = os.path.splitext(__file__)[0]
    parser = config.parse_config(script_name)
    # parser.add_argument("--a", help="some option") # additional args here
    args = parser.parse_args()
    # config.configure_logging(args)
    return args

if __name__ == "__main__":
    args = configure()
    print (args.learning_rate)
    train(args)


