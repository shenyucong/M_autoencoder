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
    tf.summary.image('input',x, 10)

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        tf.summary.histogram("weights", W_conv1)
        tf.summary.histogram("biases", b_conv1)
        tf.summary.histogram("activate", h_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        tf.summary.histogram("weights", W_conv1)
        tf.summary.histogram("biases", b_conv1)
        tf.summary.histogram("activate", h_conv2)

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
        fold1 = tf.reshape(decode, (-1, 7, 7, 32))

    with tf.name_scope('up_pool1'):
        W_up1 = weight_variable([2, 2, 32, 32])
        h_uppool1 = tf.nn.conv2d_transpose(fold1, W_up1, output_shape = [1, 14, 14, 32], strides = [1, 2, 2, 1], padding = 'VALID')

    with tf.name_scope('deconv1'):
        W_conv3 = weight_variable([5, 5, 32, 32])
        b_conv3 = bias_variable([32])
        h_deconv1 = tf.nn.relu(tf.nn.conv2d_transpose(h_uppool1, W_conv3, output_shape = [1, 14, 14, 32], strides = [1, 1, 1, 1], padding = 'SAME') + b_conv3)
        tf.summary.histogram("weights", W_conv3)
        tf.summary.histogram("biases", b_conv3)
        tf.summary.histogram("activate", h_deconv1)

    with tf.name_scope('up_pool2'):
        W_up2 = weight_variable([2, 2, 32, 32])
        h_uppool2 = tf.nn.conv2d_transpose(h_deconv1, W_up2, output_shape = [1, 28, 28, 32], strides = [1, 2, 2, 1], padding = 'VALID')

    with tf.name_scope('deconv2'):
        W_conv4 = weight_variable([5, 5, 1, 32])
        b_conv4 = bias_variable([1])
        reconstruct = tf.nn.relu(tf.nn.conv2d_transpose(h_uppool2, W_conv4, output_shape = [1, 28, 28, 1], strides = [1, 1, 1, 1], padding = 'SAME') + b_conv4) #maybe has bug
        tf.summary.histogram("weights", W_conv4)
        tf.summary.histogram("biases", b_conv4)
        tf.summary.histogram("activate", reconstruct)
    return reconstruct

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
        'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [height, width])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.expand_dims(image, -1)
    image = tf.image.resize_images(image, (28, 28))
    return image

BATCH_SIZE = 1
tfrecords_name = 'mnist.tfrecords'
tfrecords_path = '/Users/chenyucong/Desktop/research/M_autoencoder/'
train_log_dir = '/Users/chenyucong/Desktop/research/M_autoencoder/log/'
filename = os.path.join(tfrecords_path, tfrecords_name)
with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])
    images = read_and_decode(filename_queue)

    images_batch, images_useless  = tf.train.shuffle_batch([images, images], batch_size=BATCH_SIZE, num_threads=1,capacity=1000 + 3 * BATCH_SIZE, min_after_dequeue = 1000)

x = tf.placeholder(tf.float32, [None, 28, 28, 1], name = "x")
tf.summary.image('input', x, 10)

reconstruct = deep(x)
tf.summary.image('reconstruct', reconstruct, 10)

with tf.name_scope('loss'):
    loss = tf.nn.l2_loss(x - reconstruct)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

summ = tf.summary.merge_all()
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir)
    tra_summary_writer.add_graph(sess.graph)
    for i in range(20000):
        images = sess.run(images_batch)
        los, _ = sess.run([loss, train_step], feed_dict = {x: images})
        if i % 100 == 0:
            print('Step %d, loss %.4f' % (i, los))
        if i % 1000 == 0:
            checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)
            summary = sess.run(summ, feed_dict = {x: images})
        tra_summary_writer.add_summary(summary, i)
    coord.request_stop()
    coord.join(threads)

