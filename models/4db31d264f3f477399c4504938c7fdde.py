import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    data            = tf.placeholder(tf.float32,  shape = (None, 260, 260, 3), name = 'data')
    batch_num = tf.shape(data)[0]
    conv2d_0        = convolution(data, group=1, strides=[1, 1], padding='SAME', name='conv2d_0')
    conv2d_0_bn     = batch_normalization(conv2d_0, variance_epsilon=0.0010000000474974513, name='conv2d_0_bn')
    conv2d_0_activation = tf.nn.relu(conv2d_0_bn, name = 'conv2d_0_activation')
    maxpool2d_0     = tf.nn.max_pool(conv2d_0_activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='maxpool2d_0')
    conv2d_1        = convolution(maxpool2d_0, group=1, strides=[1, 1], padding='SAME', name='conv2d_1')
    conv2d_1_bn     = batch_normalization(conv2d_1, variance_epsilon=0.0010000000474974513, name='conv2d_1_bn')
    conv2d_1_activation = tf.nn.relu(conv2d_1_bn, name = 'conv2d_1_activation')
    maxpool2d_1     = tf.nn.max_pool(conv2d_1_activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='maxpool2d_1')
    conv2d_2        = convolution(maxpool2d_1, group=1, strides=[1, 1], padding='SAME', name='conv2d_2')
    conv2d_2_bn     = batch_normalization(conv2d_2, variance_epsilon=0.0010000000474974513, name='conv2d_2_bn')
    conv2d_2_activation = tf.nn.relu(conv2d_2_bn, name = 'conv2d_2_activation')
    maxpool2d_2     = tf.nn.max_pool(conv2d_2_activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='maxpool2d_2')
    conv2d_3        = convolution(maxpool2d_2, group=1, strides=[1, 1], padding='SAME', name='conv2d_3')
    conv2d_3_bn     = batch_normalization(conv2d_3, variance_epsilon=0.0010000000474974513, name='conv2d_3_bn')
    conv2d_3_activation = tf.nn.relu(conv2d_3_bn, name = 'conv2d_3_activation')
    maxpool2d_3     = tf.nn.max_pool(conv2d_3_activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='maxpool2d_3')
    cls_0_insert_conv2d = convolution(conv2d_3_activation, group=1, strides=[1, 1], padding='SAME', name='cls_0_insert_conv2d')
    loc_0_insert_conv2d = convolution(conv2d_3_activation, group=1, strides=[1, 1], padding='SAME', name='loc_0_insert_conv2d')
    conv2d_4        = convolution(maxpool2d_3, group=1, strides=[1, 1], padding='SAME', name='conv2d_4')
    cls_0_insert_conv2d_bn = batch_normalization(cls_0_insert_conv2d, variance_epsilon=0.0010000000474974513, name='cls_0_insert_conv2d_bn')
    loc_0_insert_conv2d_bn = batch_normalization(loc_0_insert_conv2d, variance_epsilon=0.0010000000474974513, name='loc_0_insert_conv2d_bn')
    conv2d_4_bn     = batch_normalization(conv2d_4, variance_epsilon=0.0010000000474974513, name='conv2d_4_bn')
    cls_0_insert_conv2d_activation = tf.nn.relu(cls_0_insert_conv2d_bn, name = 'cls_0_insert_conv2d_activation')
    loc_0_insert_conv2d_activation = tf.nn.relu(loc_0_insert_conv2d_bn, name = 'loc_0_insert_conv2d_activation')
    conv2d_4_activation = tf.nn.relu(conv2d_4_bn, name = 'conv2d_4_activation')
    cls_0_conv      = convolution(cls_0_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='cls_0_conv')
    loc_0_conv      = convolution(loc_0_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='loc_0_conv')
    maxpool2d_4     = tf.nn.max_pool(conv2d_4_activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='maxpool2d_4')
    cls_1_insert_conv2d = convolution(conv2d_4_activation, group=1, strides=[1, 1], padding='SAME', name='cls_1_insert_conv2d')
    loc_1_insert_conv2d = convolution(conv2d_4_activation, group=1, strides=[1, 1], padding='SAME', name='loc_1_insert_conv2d')
    cls_0_reshape   = tf.reshape(cls_0_conv, [batch_num, -1, 2], 'cls_0_reshape')
    loc_0_reshape   = tf.reshape(loc_0_conv, [batch_num, -1, 4], 'loc_0_reshape')
    conv2d_5        = convolution(maxpool2d_4, group=1, strides=[1, 1], padding='SAME', name='conv2d_5')
    cls_1_insert_conv2d_bn = batch_normalization(cls_1_insert_conv2d, variance_epsilon=0.0010000000474974513, name='cls_1_insert_conv2d_bn')
    loc_1_insert_conv2d_bn = batch_normalization(loc_1_insert_conv2d, variance_epsilon=0.0010000000474974513, name='loc_1_insert_conv2d_bn')
    cls_0_activation = tf.sigmoid(cls_0_reshape, name = 'cls_0_activation')
    conv2d_5_bn     = batch_normalization(conv2d_5, variance_epsilon=0.0010000000474974513, name='conv2d_5_bn')
    cls_1_insert_conv2d_activation = tf.nn.relu(cls_1_insert_conv2d_bn, name = 'cls_1_insert_conv2d_activation')
    loc_1_insert_conv2d_activation = tf.nn.relu(loc_1_insert_conv2d_bn, name = 'loc_1_insert_conv2d_activation')
    conv2d_5_activation = tf.nn.relu(conv2d_5_bn, name = 'conv2d_5_activation')
    cls_1_conv      = convolution(cls_1_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='cls_1_conv')
    loc_1_conv      = convolution(loc_1_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='loc_1_conv')
    maxpool2d_5     = tf.nn.max_pool(conv2d_5_activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='maxpool2d_5')
    cls_2_insert_conv2d = convolution(conv2d_5_activation, group=1, strides=[1, 1], padding='SAME', name='cls_2_insert_conv2d')
    loc_2_insert_conv2d = convolution(conv2d_5_activation, group=1, strides=[1, 1], padding='SAME', name='loc_2_insert_conv2d')
    cls_1_reshape   = tf.reshape(cls_1_conv, [batch_num, -1, 2], 'cls_1_reshape')
    loc_1_reshape   = tf.reshape(loc_1_conv, [batch_num, -1, 4], 'loc_1_reshape')
    conv2d_6        = convolution(maxpool2d_5, group=1, strides=[1, 1], padding='SAME', name='conv2d_6')
    cls_2_insert_conv2d_bn = batch_normalization(cls_2_insert_conv2d, variance_epsilon=0.0010000000474974513, name='cls_2_insert_conv2d_bn')
    loc_2_insert_conv2d_bn = batch_normalization(loc_2_insert_conv2d, variance_epsilon=0.0010000000474974513, name='loc_2_insert_conv2d_bn')
    cls_1_activation = tf.sigmoid(cls_1_reshape, name = 'cls_1_activation')
    conv2d_6_bn     = batch_normalization(conv2d_6, variance_epsilon=0.0010000000474974513, name='conv2d_6_bn')
    cls_2_insert_conv2d_activation = tf.nn.relu(cls_2_insert_conv2d_bn, name = 'cls_2_insert_conv2d_activation')
    loc_2_insert_conv2d_activation = tf.nn.relu(loc_2_insert_conv2d_bn, name = 'loc_2_insert_conv2d_activation')
    conv2d_6_activation = tf.nn.relu(conv2d_6_bn, name = 'conv2d_6_activation')
    cls_2_conv      = convolution(cls_2_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='cls_2_conv')
    loc_2_conv      = convolution(loc_2_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='loc_2_conv')
    conv2d_7        = convolution(conv2d_6_activation, group=1, strides=[1, 1], padding='VALID', name='conv2d_7')
    cls_3_insert_conv2d = convolution(conv2d_6_activation, group=1, strides=[1, 1], padding='SAME', name='cls_3_insert_conv2d')
    loc_3_insert_conv2d = convolution(conv2d_6_activation, group=1, strides=[1, 1], padding='SAME', name='loc_3_insert_conv2d')
    cls_2_reshape   = tf.reshape(cls_2_conv, [batch_num, -1, 2], 'cls_2_reshape')
    loc_2_reshape   = tf.reshape(loc_2_conv, [batch_num, -1, 4], 'loc_2_reshape')
    conv2d_7_bn     = batch_normalization(conv2d_7, variance_epsilon=0.0010000000474974513, name='conv2d_7_bn')
    cls_3_insert_conv2d_bn = batch_normalization(cls_3_insert_conv2d, variance_epsilon=0.0010000000474974513, name='cls_3_insert_conv2d_bn')
    loc_3_insert_conv2d_bn = batch_normalization(loc_3_insert_conv2d, variance_epsilon=0.0010000000474974513, name='loc_3_insert_conv2d_bn')
    cls_2_activation = tf.sigmoid(cls_2_reshape, name = 'cls_2_activation')
    conv2d_7_activation = tf.nn.relu(conv2d_7_bn, name = 'conv2d_7_activation')
    cls_3_insert_conv2d_activation = tf.nn.relu(cls_3_insert_conv2d_bn, name = 'cls_3_insert_conv2d_activation')
    loc_3_insert_conv2d_activation = tf.nn.relu(loc_3_insert_conv2d_bn, name = 'loc_3_insert_conv2d_activation')
    cls_4_insert_conv2d = convolution(conv2d_7_activation, group=1, strides=[1, 1], padding='SAME', name='cls_4_insert_conv2d')
    loc_4_insert_conv2d = convolution(conv2d_7_activation, group=1, strides=[1, 1], padding='SAME', name='loc_4_insert_conv2d')
    cls_3_conv      = convolution(cls_3_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='cls_3_conv')
    loc_3_conv      = convolution(loc_3_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='loc_3_conv')
    cls_4_insert_conv2d_bn = batch_normalization(cls_4_insert_conv2d, variance_epsilon=0.0010000000474974513, name='cls_4_insert_conv2d_bn')
    loc_4_insert_conv2d_bn = batch_normalization(loc_4_insert_conv2d, variance_epsilon=0.0010000000474974513, name='loc_4_insert_conv2d_bn')
    cls_3_reshape   = tf.reshape(cls_3_conv, [batch_num, -1, 2], 'cls_3_reshape')
    loc_3_reshape   = tf.reshape(loc_3_conv, [batch_num, -1, 4], 'loc_3_reshape')
    cls_4_insert_conv2d_activation = tf.nn.relu(cls_4_insert_conv2d_bn, name = 'cls_4_insert_conv2d_activation')
    loc_4_insert_conv2d_activation = tf.nn.relu(loc_4_insert_conv2d_bn, name = 'loc_4_insert_conv2d_activation')
    cls_3_activation = tf.sigmoid(cls_3_reshape, name = 'cls_3_activation')
    cls_4_conv      = convolution(cls_4_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='cls_4_conv')
    loc_4_conv      = convolution(loc_4_insert_conv2d_activation, group=1, strides=[1, 1], padding='SAME', name='loc_4_conv')
    cls_4_reshape   = tf.reshape(cls_4_conv, [batch_num, -1, 2], 'cls_4_reshape')
    loc_4_reshape   = tf.reshape(loc_4_conv, [batch_num, -1, 4], 'loc_4_reshape')
    cls_4_activation = tf.sigmoid(cls_4_reshape, name = 'cls_4_activation')
    loc_branch_concat = tf.concat([loc_0_reshape, loc_1_reshape, loc_2_reshape, loc_3_reshape, loc_4_reshape], 1, name = 'loc_branch_concat')
    cls_branch_concat = tf.concat([cls_0_activation, cls_1_activation, cls_2_activation, cls_3_activation, cls_4_activation], 1, name = 'cls_branch_concat')
    return data, loc_branch_concat, cls_branch_concat


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, name=name, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, name=name, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer

def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)


