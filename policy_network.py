import numpy as np
import tensorflow as tf

from constants import LEARNING_RATE


class PolicyNetwork(object):
    def __init__(self, n_actions, input_shape):
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.optimizer = None
        self.sess = tf.Session()
        self._build_model()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, batch):
        assert self.optimizer and batch

        states, rewards = [], []
        for data_point in batch:
            states.append(data_point['current_state'])
            rewards.append(data_point['reward'])

        self.sess.run([self.optimizer], feed_dict={self.x: np.asarray(states), self.reward: np.asarray(rewards)})

    def predict(self, state):
        state = state.astype(np.float32)
        return self.sess.run([self.out], feed_dict={self.x: state})

    def _build_model(self):
        self._init_weights_and_biases()
        self.x = tf.placeholder(tf.float32, shape=(None, *self.input_shape))
        self.reward = tf.placeholder(tf.float32, shape=(None, 1))     # sum of discounted rewards

        conv1_out = tf.nn.conv2d(input=self.x, filter=self.weights['conv1'], strides=[1, 4, 4, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(value=conv1_out, bias=self.biases['Bconv1']))
        conv2_out = tf.nn.conv2d(input=conv1, filter=self.weights['conv2'], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(value=conv2_out, bias=self.biases['Bconv2']))
        conv3_out = tf.nn.conv2d(input=conv2, filter=self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.relu(tf.nn.bias_add(value=conv3_out, bias=self.biases['Bconv3']))
        flattened_conv3 = tf.reshape(conv3, [-1, self.weights['fc'].get_shape().as_list()[0]])
        print('flattened_conv3 ', flattened_conv3.get_shape().as_list())
        fc = tf.nn.relu(tf.add(tf.matmul(flattened_conv3, self.weights['fc']), self.biases['Bfc']))
        self.out = tf.nn.softmax(tf.matmul(fc, self.weights['out']))
        print('Out Shape ', tf.log(self.out).get_shape().as_list())

        trainable_params = [self.weights['conv1'], self.weights['conv2'], self.weights['conv3'], self.weights['fc'],
                            self.weights['out'], self.biases['Bconv1'], self.biases['Bconv2'],
                            self.biases['Bconv3'], self.biases['Bfc']]
        gradient = tf.gradients(ys=tf.log(self.out), xs=trainable_params, name='characteristic_eligibility')
        print('Gradient len ', len(gradient))
        print('one Gradient Shape ', gradient[0].get_shape().as_list())
        print('Reward Shape ', self.reward.get_shape())
        self.cost = tf.multiply(self.reward, np.asarray(gradient), name='cost')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(-1 * self.cost)

    def _init_weights_and_biases(self):
        self.weights = {
            'conv1': tf.Variable(tf.random_normal([8, 8, 4, 32]), name='conv1'),
            'conv2': tf.Variable(tf.random_normal([8, 8, 32, 64]), name='conv2'),
            'conv3': tf.Variable(tf.random_normal([3, 3, 64, 64]), name='conv3'),
            # For 84x84 input and 3 conv layer with filter size of 8x8, 8x8, 3x3 and stride 4, 2, 1 number of neurons
            #  in 3rd conv layer will be 5x5
            # # neurons = ((i/p dimension - filter size + 2*padding size)/stride) + 1
            # http://cs231n.github.io/convolutional-networks/#conv
            'fc': tf.Variable(tf.random_normal([5*5, 512]), name='fc'),
            'out': tf.Variable(tf.random_normal([512, self.n_actions]), name='out')
        }
        self.biases = {
            'Bconv1': tf.Variable(tf.random_normal([32]), name='Bconv1'),
            'Bconv2': tf.Variable(tf.random_normal([64]), name='Bconv2'),
            'Bconv3': tf.Variable(tf.random_normal([64]), name='Bconv3'),
            'Bfc': tf.Variable(tf.random_normal([512]), name='Bfc')
        }
