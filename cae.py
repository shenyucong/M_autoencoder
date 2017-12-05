import tensorflow as tf

def deep(x):
    '''
    Build the convolutional auto-encoder
    Args:
        x: the input images, one image per time
    Returns:
        the logits
    '''
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        tf.summary.histogram("weights", W_conv1)
        tf.summary.histogram("biases", b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variables([5, 5, 32, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        tf.summary.histogram("weights", W_conv1)
        tf.summary.histogram("biases", b_conv1)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool(h_conv2)

    with tf.name_scope('unfold'):
        num, height, width, channels = h_pool2.getshape()
        unfold1 = tf.reshape(h_pool2, (-1, height, width, channels))

    with tf.name_scope('encode'):
        W_fc1 = weight_variable([7*7*32, 20])

def conv2d(x, W):
    '''conv2d returns a 2d convolution layer with full stride'''
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    '''max_pool downsamples a feature map by 2x.'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def weight_variable(shape):
    '''weight_variable returns a weights variable of a given shape'''
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = "W")

def bias_variable(shape):
    '''bias_variable returns a bias variable of a given shape'''
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = "B")

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [height, width])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    image = tf.expand_dims(image, -1)
    image = tf.image.resize_images(image, (28, 28))
    label = tf.one_hot(tf.cast(label, tf.int32), depth = 10)
    return image, label

def num_correct_prediction(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
        the number of correct predictions
    """
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct
