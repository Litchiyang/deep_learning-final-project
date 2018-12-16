import tensorflow as tf
import numpy as np
import os
import config
import gan
import readfile

def train(dataset_path, batch_size, model_root, d_learning_rate, g_learning_rate, iteration, image_size, dim, istrain=True):
    print("start training..")
    
    #read data
    data_cele = readfile.CelebA(dataset_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)   #create a session

    #create an GAN object
    gan_model = gan.GAN(image_size, batch_size, dim, istrain) 

    #build a graph and get the loss
    z,d,D_loss, G_loss = gan_model.build_model()

    #optimizer 
    theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis_')
    d_optimizer = tf.train.AdamOptimizer(learning_rate = d_learning_rate).minimize(D_loss, var_list=theta_D)
    
    theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen_')
    g_optimizer = tf.train.AdamOptimizer(learning_rate = g_learning_rate).minimize(G_loss, var_list=theta_G)
    
    #initialize
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    
    # summary
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(model_root, "logs/"), session.graph)
    
    for i in range(iteration):
        

        _, D_loss_curr, D_summary = session.run([d_optimizer, D_loss, summary_op], 
            feed_dict={z: sample_z(batch_size, dim), d: data_cele.train_next_batch(batch_size, image_size)})
        _, G_loss_curr = session.run([g_optimizer, G_loss], 
            feed_dict={z: sample_z(batch_size, dim), d: data_cele.train_next_batch(batch_size, image_size)})
        _, G_loss_curr = session.run([g_optimizer, G_loss], 
            feed_dict={z: sample_z(batch_size, dim), d: data_cele.train_next_batch(batch_size, image_size)})
        print("iter: ", i, " finished, D loss: ", D_loss_curr, ", G loss: ", G_loss_curr)
        writer.add_summary(D_summary, i)
        if i % 100 == 0:
            save_tf_model(session, model_root, saver, i, pb=True)

    print("Training finished!")
        
        

def sample_z(batch_size, dim):
    return np.random.uniform(-1., 1., size=[batch_size, dim])


def save_tf_model(sess, save_root, saver, i="", pb=False):
    if i:
        ckpt_path = "model_{}.ckpt".format(i)
    else:
        ckpt_path = "model.ckpt"
    ckpt_path = os.path.join(save_root, ckpt_path)
    saver.save(sess, ckpt_path)
    
    graph_meta_file = os.path.join(save_root, 'graph_def.meta')
    # save 
    if pb:
        tf.train.export_meta_graph(filename=graph_meta_file)
        graph_folder = os.path.join(save_root, 'graph')
        graph_pb_name = 'graph.pb'
        tf.train.write_graph(sess.graph_def, graph_folder, graph_pb_name, False)
