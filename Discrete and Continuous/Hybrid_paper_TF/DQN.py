import os
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.compat.v1.initializers import random_uniform
import tensorflow.compat.v1.keras.backend as tf_k



class ActorDQN(object):
    def __init__(self, lr, n_discrete_actions, n_continuous_actions, name, input_dims, sess, fc1_dims, fc2_dims,
                 fc3_dims, batch_size=32, chkpt_dir='./tmp/DQN/'):
        self.lr = lr
        self.n_discrete_actions = n_discrete_actions
        self.n_continuous_actions = n_continuous_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name +'_ddpg.ckpt')

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.continuous_action_gradients = tf.gradients(self.Q, self.actions_continuous)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')

            self.actions_continuous = tf.placeholder(tf.float32,
                                          shape=[None, self.n_continuous_actions],
                                          name='actions')

            self.Qvalues_target = tf.placeholder(tf.float32,
                                           shape=[None, self.n_discrete_actions],
                                           name='Qvalues')

            inputs = tf.concat([self.input, self.actions_continuous], axis=1, name = "DQN_concat")
            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(inputs, units=self.fc1_dims,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1),
                                     name="DQN_dense1")
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1. / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            f3 = 1. / np.sqrt(self.fc3_dims)
            dense3 = tf.layers.dense(layer2_activation, units=self.fc3_dims,
                                     kernel_initializer=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3, f3))
            batch3 = tf.layers.batch_normalization(dense3)
            layer3_activation = tf.nn.relu(batch3)


            f4 = 0.003
            self.Qvalues = tf.layers.dense(layer3_activation, units=self.n_discrete_actions,
                               kernel_initializer=random_uniform(-f4, f4),
                               bias_initializer=random_uniform(-f4, f4),
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))

            self.Q = tf_k.sum(self.Qvalues)

            self.loss = tf.losses.mean_squared_error(self.Qvalues_target, self.Qvalues)

    def predict(self, inputs, actions_continuous):
        return self.sess.run(self.Qvalues,
                             feed_dict={self.input: inputs,
                                        self.actions_continuous: actions_continuous})

    def train(self, inputs, actions_continuous, Qvalues_target):
        return self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.actions_continuous: actions_continuous,
                                 self.Qvalues_target: Qvalues_target})

    def get_action_gradients(self, inputs, actions_continuous):
        return self.sess.run(self.continuous_action_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions_continuous: actions_continuous})
    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)