import tensorflow as tf
import os.path

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
        unfold1 = tf.reshape(h_pool2, (-1, 7*7*32))

    with tf.name_scope('encode'):
        W_fc1 = weight_variable([7*7*32, 20])
        b_fc1 = bias_variable([20])

        encode = tf.nn.relu(tf.matmul(unfold1, W_fc1) + b_fc1)
        tf.summary.histogram("weights", W_fc1)
        tf.summary.histogram("biases", b_fc1)
        tf.summary.histogram("encode/relu", encode)

    with tf.name_scope('decode'):
        W_fc2 = weight_variable([20, 7*7*32])
        b_fc2 = bias_variable([7*7*32])
        decode = tf.nn.relu(tf.matmul(encode, W_fc2) + b_fc2)
        tf.summary.histogram("weights", W_fc2)
        tf.summary.histogram("biases", b_fc2)
        tf.summary.histogram("decode/relu", decode)

    with tf.name_scope('fold'):
        fold1 = tf.reshpae(decode, (-1, 7, 7, 32))

    with tf.name_scope('up_pool1'):
        h_uppool1 = up_pool(fold1) #how to implement up_pool?

    with tf.name_scope('deconv1'):
        W_conv3 = weight_variables([5, 5, 32, 32])
        b_conv3 = bias_variable([32])
        h_deconv1 = tf.nn.relu(deconv2d(h_uppool1, W_conv3) + b_conv3)
        tf.summary.histogram("weights", W_conv3)
        tf.summary.histogram("biases", b_conv3)

    with tf.name_scope('up_pool2'):
        h_uppool2 = up_pool(h_deconv1)

    with tf.name_scope('deconv2'):
        W_conv4 = weight_variables([5, 5, 1, 32])
        b_conv4 = bias_variable([1])
        reconstruct = tf.nn.relu(deconv2d(h_uppool2, W_conv4) + b_conv4) #maybe has bug
        tf.summary.histogram("weights", W_conv4)
        tf.summary.histogram("biases", b_conv4)

def conv2d(x, W):
    '''conv2d returns a 2d convolution layer with full stride'''
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    '''max_pool downsamples a feature map by 2x.'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def deconv2d(x, W):
    '''deconv2d returns a 2d transpose convolution layer with full stride'''
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def up_pool(x):
    '''up_pool returns upsamples a feature map by 2x.'''
    return tf.nn.conv2d_transpose(x, [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID')

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

BATCHE_SIZE = 25
tfrecords_name = ''
tfrecords_path = ''
filename = os.path.join(tfrecords_path, tfrecords_name)
with name_scope('input'):
    images = read_and_decode(filename_queue)
    images_batch  = tf.train.shuffle_batch(images, batch_size=BATCH_SIZE, num_threads=1,capacity=1000 + 3 * BATCH_SIZE, min_after_dequeue = 1000)

x = tf.placeholder(tf.float32, [None, 28, 28, 1], name = "x")
tf.summary.image('input', x, 3)

y_ = tf.placeholder(tf.float32, [None, 18], name = "labels")

y_conv, keep_prob = deepnn(x)

with name_scope('loss'):

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

summ = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir)
    tra_summary_writer.add_graph(sess.graph)
    images = sess.run([images_batch])
    print(images)
    summary = sess.run([summ])
    tra_summary_writer.add_summary(summary, i)
    coord.request_stop()
    coord.join(threads)

