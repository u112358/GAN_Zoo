#!/usr/bin/env python


import tensorflow as tf


def encoder(inputs, z_dim=128, n_layers=4, kernel_size=[7, 5, 5, 3], filters=[32, 64, 128, 256],
            strides=[2, 2, 2, (2, 1)], with_bn=[True, True, True, True], reuse=False, scope_name='encoder'):
    with tf.variable_scope(scope_name):
        current = inputs
        for n in range(n_layers):
            name = 'conv' + str(n + 1)
            current = tf.layers.conv2d(inputs=current, filters=filters[n], kernel_size=kernel_size[n],
                                       strides=strides[n], padding='SAME', use_bias=True, reuse=reuse, name=name,
                                       kernel_initializer=tf.random_normal_initializer(stddev=1e-2))
            current = tf.nn.relu(current)
            if with_bn[n]:
                name = 'bn' + str(n + 1)
                current = tf.layers.batch_normalization(current, reuse=reuse, name=name)

        input_shape = inputs.get_shape().as_list()
        height = input_shape[1]
        width = input_shape[2]

        for n in range(n_layers):
            if isinstance(strides[n], tuple):
                height = height / strides[n][0]
                width = width / strides[n][1]
            else:
                height = height / strides[n]
                width = width / strides[n]
        current = tf.reshape(current, (-1, int(height * width * filters[n_layers - 1])))
        current = tf.contrib.layers.fully_connected(current, z_dim, reuse=reuse, scope='fc')

        return current


def encoder_old_school(inputs, z_dim=128, reuse=False, scope_name='encoder'):
    with tf.variable_scope(scope_name):
        current = inputs
        current = tf.layers.conv2d(inputs=current, filters=16, kernel_size=7, strides=1, padding='SAME', use_bias=True,
                                   reuse=reuse, name='conv1')
        current = tf.layers.max_pooling2d(inputs=current, pool_size=2, strides=2, name='pool1')
        current = tf.nn.relu(current)
        current = tf.layers.conv2d(inputs=current, filters=64, kernel_size=7, strides=1, padding='SAME', use_bias=True,
                                   reuse=reuse, name='conv2')
        current = tf.layers.max_pooling2d(inputs=current, pool_size=2, strides=2, name='pool2')
        current = tf.nn.relu(current)
        current = tf.layers.conv2d(inputs=current, filters=256, kernel_size=7, strides=1, padding='SAME', use_bias=True,
                                   reuse=reuse, name='conv3')
        current = tf.nn.relu(current)
        current = tf.layers.batch_normalization(inputs=current, reuse=reuse, name='bn1')
        current = tf.layers.dropout(inputs=current, rate=0.5, name='dp')
        current = tf.layers.flatten(current)
        current = tf.layers.dense(inputs=current, units=z_dim, reuse=reuse)
        return current


def encoder_deep_cnn(inputs, z_dim=128, reuse=False, scope_name='encoder'):
    with tf.variable_scope(scope_name):
        current = inputs
        current = tf.layers.conv2d(inputs=current, filters=16, kernel_size=7, strides=1, padding='SAME', use_bias=True,
                                   reuse=reuse, name='conv1')
        current = tf.nn.relu(current)
        current = tf.nn.local_response_normalization(current, depth_radius=5, bias=2, alpha=1e-4, beta=0.75,
                                                     name='lrn1')
        current = tf.layers.max_pooling2d(current, strides=2, pool_size=2, name='pool1')

        current = tf.layers.conv2d(inputs=current, filters=64, kernel_size=7, strides=1, padding='SAME', use_bias=True,
                                   reuse=reuse, name='conv2')
        current = tf.nn.relu(current)
        current = tf.nn.local_response_normalization(current, depth_radius=5, bias=12, alpha=1e-4, beta=0.75,
                                                     name='lrn2')
        current = tf.layers.max_pooling2d(current, strides=2, pool_size=2, name='pool2')

        current = tf.layers.conv2d(inputs=current, filters=256, kernel_size=7, strides=1, padding='SAME', use_bias=True,
                                   reuse=reuse, name='conv3')
        current = tf.layers.dropout(inputs=current, rate=0.5, name='dp')
        current = tf.layers.flatten(current)
        current = tf.layers.dense(inputs=current, units=z_dim, reuse=reuse)
        return current


def generator(inputs, width=88, height=128, init_channel=256, labels=None, n_layers=4,
              kernel_size=[4, 4, 4, 4],
              filters=[128, 64, 32, 1], strides=[(2, 1), 2, 2, 2], with_bn=[True, True, True, True], reuse=False,
              scope_name='generator'):
    with tf.variable_scope(scope_name):
        current = inputs
        for n in range(n_layers):
            if isinstance(strides[n], tuple):
                height = height / strides[n][0]
                width = width / strides[n][1]
            else:
                height = height / strides[n]
                width = width / strides[n]

        if not (labels is None):
            current = tf.concat([current, labels], axis=1)
        current = tf.contrib.layers.fully_connected(current, int(width * height * init_channel))
        current = tf.nn.relu(current)
        current = tf.reshape(current, (-1, int(height), int(width), init_channel))

        for n in range(n_layers):
            name = 'deconv' + str(n + 1)
            current = tf.layers.conv2d_transpose(inputs=current, filters=filters[n], kernel_size=kernel_size[n],
                                                 strides=strides[n], padding='SAME', use_bias=True, reuse=reuse,
                                                 name=name,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            if n == n_layers - 1:
                current = tf.nn.tanh(current)
            else:
                current = tf.nn.relu(current)
                if with_bn[n]:
                    name = 'bn' + str(n + 1)
                    current = tf.layers.batch_normalization(current, reuse=reuse, name=name)

        return current


def pairwise_discriminator(inputs, target, n_layers=4, kernel_size=[7, 5, 5, 3], filters=[32, 64, 128, 256],
                           strides=[2, 2, 2, (2, 1)], with_bn=[True, True, True, True], reuse=False,
                           scope_name='discriminator'):
    with tf.variable_scope(scope_name):
        current = tf.concat([inputs, target], axis=3)
        for n in range(n_layers):
            name = 'conv' + str(n + 1)
            current = tf.layers.conv2d(inputs=current, filters=filters[n], kernel_size=kernel_size[n],
                                       strides=strides[n], padding='SAME', use_bias=True, reuse=reuse, name=name,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            current = tf.nn.relu(current)
            if with_bn[n]:
                name = 'bn' + str(n + 1)
                current = tf.layers.batch_normalization(current, reuse=reuse, name=name)

        input_shape = inputs.get_shape().as_list()
        height = input_shape[1]
        width = input_shape[2]

        for n in range(n_layers):
            if isinstance(strides[n], tuple):
                height = height / strides[n][0]
                width = width / strides[n][1]
            else:
                height = height / strides[n]
                width = width / strides[n]
        current = tf.reshape(current, (-1, int(height * width * filters[n_layers - 1])))
        current = tf.contrib.layers.fully_connected(current, 1, reuse=reuse, scope='fc')

    return current, tf.sigmoid(current)


def conditional_discriminator(inputs, label, n_layers=4, kernel_size=[7, 5, 5, 3], filters=[32, 64, 128, 256],
                              strides=[2, 2, 2, (2, 1)], with_bn=[True, True, True, True], reuse=False,
                              scope_name='conditional_discriminator'):
    with tf.variable_scope(scope_name):
        current = inputs
        for n in range(n_layers):
            if n == 1:
                current = concat_label(current, label)
            name = 'conv' + str(n + 1)
            current = tf.layers.conv2d(inputs=current, filters=filters[n], kernel_size=kernel_size[n],
                                       strides=strides[n], padding='SAME', use_bias=True, reuse=reuse, name=name,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            current = tf.nn.relu(current)
            if with_bn[n]:
                name = 'bn' + str(n + 1)
                current = tf.layers.batch_normalization(current, reuse=reuse, name=name)

        input_shape = inputs.get_shape().as_list()
        height = input_shape[1]
        width = input_shape[2]

        for n in range(n_layers):
            if isinstance(strides[n], tuple):
                height = height / strides[n][0]
                width = width / strides[n][1]
            else:
                height = height / strides[n]
                width = width / strides[n]
        current = tf.reshape(current, (-1, int(height * width * filters[n_layers - 1])))
        current = tf.contrib.layers.fully_connected(current, 1, reuse=reuse, scope='fc')

        return current, tf.sigmoid(current)


def concat_label(x, label, duplicate=1):
    x_shape = x.get_shape().as_list()
    if duplicate < 1:
        return x
    label = tf.tile(label, [1, duplicate])
    label_shape = label.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat(axis=1, values=[x, label])
    elif len(x_shape) == 4:
        label = tf.reshape(label, [x_shape[0], 1, 1, label_shape[-1]])
        return tf.concat(axis=3, values=[x, label * tf.ones([x_shape[0], x_shape[1], x_shape[2], label_shape[-1]])])


