from GANlib.layers import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random


class ConDCGAN:
    def __init__(self):
        self.img_width = 28
        self.img_height = 28
        self.img_channel = 1
        self.z_dims = 32
        self.nof_classes = 10
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = 1e-5
        self.train_epoch = 100
        self.batch_size = 50
        self.input_img = tf.placeholder(shape=(self.batch_size, self.img_height, self.img_width, self.img_channel),
                                        dtype=tf.float32,
                                        name='input_img')
        self.input_label = tf.placeholder(shape=(self.batch_size, self.nof_classes), dtype=tf.float32,
                                          name='input_label')
        self.input_z = tf.placeholder(shape=(self.batch_size, self.z_dims), dtype=tf.float32, name='input_z')
        self.generated_img = generator(inputs=self.input_z, labels=self.input_label, width=self.img_width,
                                       height=self.img_height, n_layers=3, filters=[256, 128, 1],
                                       kernel_size=[7, 5, 5],
                                       strides=[1, 2, 2], with_bn=[False, False, False], reuse=False,
                                       scope_name='generator')
        D_true_logits, D_true = conditional_discriminator(self.input_img, self.input_label, n_layers=3,
                                                          kernel_size=[5, 5, 7], strides=[2, 2, 1],
                                                          with_bn=[False, False, False], reuse=False,
                                                          scope_name='discriminator')
        D_false_logits, D_false = conditional_discriminator(self.generated_img, self.input_label, n_layers=3,
                                                            kernel_size=[5, 5, 7], strides=[2, 2, 1],
                                                            with_bn=[False, False, False], reuse=tf.AUTO_REUSE,
                                                            scope_name='discriminator')
        G_true_logits, G_true = conditional_discriminator(self.generated_img, self.input_label, n_layers=3,
                                                          kernel_size=[5, 5, 7], strides=[2, 2, 1],
                                                          with_bn=[False, False, False], reuse=tf.AUTO_REUSE,
                                                          scope_name='discriminator')
        D_loss_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=D_true_logits, labels=tf.ones_like(D_true_logits)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false_logits, labels=tf.zeros_like(D_false_logits)))
        self.D_loss = D_loss_fake + D_loss_real
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=G_true_logits, labels=tf.ones_like(G_true_logits)))
        vars = tf.trainable_variables()
        D_vars = [v for v in vars if v.name.startswith('discriminator')]
        G_vars = [v for v in vars if v.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optim = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            self.D_optim = optim.minimize(self.D_loss, global_step=self.global_step, var_list=D_vars)
            # D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
            self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.G_loss, var_list=G_vars)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
            train_set = (mnist.train.images - 0.5) / 0.5
            train_label = mnist.train.labels
            for epoch in range(self.train_epoch):
                shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
                shuffled_set = train_set[shuffle_idxs]
                shuffled_label = train_label[shuffle_idxs]
                for iter in range(shuffled_set.shape[0] // self.batch_size):
                    # update discriminator
                    x_ = shuffled_set[iter * self.batch_size:(iter + 1) * self.batch_size]
                    y_label_ = shuffled_label[iter * self.batch_size:(iter + 1) * self.batch_size].reshape(
                        [self.batch_size, self.nof_classes])
                    y_label_ = np.asarray(y_label_)
                    z_ = np.random.normal(0, 1, (self.batch_size, self.z_dims))

                    loss_d_, _ = sess.run([self.D_loss, self.D_optim],
                                          {self.input_img: x_, self.input_z: z_, self.input_label: y_label_})

                    # update generator
                    # z_ = np.random.normal(0, 1, (self.batch_size, self.z_dims))
                    loss_g_, _ = sess.run([self.G_loss, self.G_optim],
                                          {self.input_label: y_label_, self.input_z: z_})


if __name__ == '__main__':
    model = ConDCGAN()
    model.train()
