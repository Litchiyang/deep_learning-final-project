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
    
    # summary
#     D_summary_op = tf.summary.merge_all(tf.GraphKeys.SUMMARIES, scope='dis_')
    summary_op = tf.summary.merge_all()
#     G_summary_op = tf.summary.merge_all(tf.GraphKeys.SUMMARIES, scope='gen_')
    writer = tf.summary.FileWriter(os.path.join(args.model_root, "logs/"), session.graph)
    
    for i in range(args.iteration):
        
        
        _, D_loss_curr, D_summary = session.run([d_optimizer, D_loss, summary_op], 
            feed_dict={z: sample_z(args.batch_size, args.dim), d: data_cele.train_next_batch(args.batch_size, args.image_size)})
        _, G_loss_curr = session.run([g_optimizer, G_loss], 
            feed_dict={z: sample_z(args.batch_size, args.dim), d: data_cele.train_next_batch(args.batch_size, args.image_size)})
        print("iter: ", i, " finished, D loss: ", D_loss_curr, ", G loss: ", G_loss_curr)
        writer.add_summary(D_summary, i)
#         writer.add_summary(G_summary, i)
        save_tf_model(session, args.model_root, saver, i, pb=True)
        
        


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
#     return ckpt_path

def configure():
    script_name = os.path.splitext(__file__)[0]
    parser = config.parse_config(script_name)
    # parser.add_argument("--a", help="some option") # additional args here
    args = parser.parse_args()
    # config.configure_logging(args)
    return args

if __name__ == "__main__":
    args = configure()
    train(args)





