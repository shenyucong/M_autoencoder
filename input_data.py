import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
import random
from skimage.transform import resize
from PIL import Image

def get_file(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        images: image directories, list, string
        labels: label, list, int
    '''

    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        #image directories
        for name in files:
            images.append(os.path.join(root, name))
        #get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root, name))

    #print(temp)
    #temp = temp.transpose()
    #np.random.shuffle(temp)
    #print(temp)

    image_list = list(images)

    return image_list

def int64_feature(value):
    '''Wrapper for inserting int64 features into Example proto.'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def convert_to_tfrecord(images, save_dir, name):
    '''
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the drectory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'

    Return:
        no return
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')

    n_samples = len(images)

    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    i = 0
    j = 0
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i], as_grey=True)
            #limage = resize(image, (28, 28))
            image_raw = image.tostring()
            height, width = image.shape
            example = tf.train.Example(features = tf.train.Features(feature={
                'height': int64_feature(height),
                'width': int64_feature(width),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it\n')
    writer.close()
    print('Transform done!\n')

def read_and_decode(tfrecords_file, batch_size):
    '''
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, with, height, channel]
        label:1D tensor - [batch_size]
    '''

    #make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'height':tf.FixedLenFeature([], tf.int64),
            'width' :tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    height = tf.cast(img_features['height'], tf.int32)
    width = tf.cast(img_features['width'], tf.int32)
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)



    image = tf.reshape(image, [height, width])
    image = tf.expand_dims(image, -1)
    image = tf.image.resize_images(image, (28, 28))
    image_batch = tf.train.shuffle_batch(image,
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = 2000,
                                                min_after_dequeue = 1000)
    #image_batch = tf.expand_dims(image_batch, -1)
    return image_batch

train_dir = '/Users/chenyucong/Desktop/research/M_autoencoder/testSet'
save_dir = '/Users/chenyucong/Desktop/research/M_autoencoder/'

name_train = 'bees_train'
images_train = get_file(train_dir)
convert_to_tfrecord(images_train, save_dir, name_train)

#def plot_images(images, labels):
#    for i in np.arange(0, 25):
#        plt.subplot(5, 5, i + 1)
#        plt.axis('off')
#        plt.title(chr(ord('A') + labels[i] - 1), fontsize = 14)
#        plt.subplots_adjust(top=1.5)
#        plt.imshow(images[i])
#    plt.show()
#with tf.Session() as sess:

#    i = 0
#j    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)

#    try:
#        while not coord.should_stop() and i<1:
            #just plot one batch size
#            image, label = sess.run([image_batch, label_batch])
#            print(image)
#            plot_images(image, label)
#            i+=1

#    except tf.errors.OutOFRangeError:
#        print('done!\n')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
