import tensorflow as tf


def get_contrastive_loss(labels, anchor, positive, margin=1.0):
    with tf.name_scope("contrastive-loss"):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(anchor - positive, 2), 1, keepdims=True))
        similarity = labels * tf.square(distance)
        dissimilarity = (1 - labels) * tf.square(tf.maximum((margin - distance), 0.0))
        return tf.reduce_mean(dissimilarity + similarity) / 2


def get_triplet_loss(embeddings, length):
    anchor = embeddings[0:length:3][:]
    positive = embeddings[1:length:3][:]
    negative = embeddings[2:length:3][:]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    with tf.name_scope('distances'):
        tf.summary.histogram('positive', pos_dist, collections=['general'])
        tf.summary.histogram('negative', neg_dist, collections=['general'])

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), 0.5)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss


def get_quartet_loss(embeddings, length):
    anchor = embeddings[0:length:4][:]
    positive = embeddings[1:length:4][:]
    negative_1 = embeddings[2:length:4][:]
    negative_2 = embeddings[3:length:4][:]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist1 = tf.reduce_sum(tf.square(tf.subtract(anchor, negative_1)), 1)
    neg_dist2 = tf.reduce_sum(tf.square(tf.subtract(anchor, negative_2)), 1)
    neg_dist3 = tf.reduce_sum(tf.square(tf.subtract(negative_1, negative_2)), 1)

    with tf.name_scope('distances'):
        tf.summary.histogram('positive', pos_dist, collections=['general'])
        tf.summary.histogram('negative_1', neg_dist1, collections=['general'])
        tf.summary.histogram('negative_2', neg_dist2, collections=['general'])
        tf.summary.histogram('negative_3', neg_dist3, collections=['general'])

    #
    basic_loss1 = tf.add(tf.subtract(pos_dist, neg_dist1), 0.5)
    loss1 = tf.reduce_mean(tf.maximum(basic_loss1, 0.0), 0)
    basic_loss2 = tf.add(tf.subtract(pos_dist, neg_dist2), 0.5)
    loss2 = tf.reduce_mean(tf.maximum(basic_loss2, 0.0), 0)
    basic_loss3 = tf.add(tf.subtract(pos_dist, neg_dist3), 0.5)
    loss3 = tf.reduce_mean(tf.maximum(basic_loss3, 0.0), 0)
    return loss1 + loss2 + loss3
