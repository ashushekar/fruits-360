import tensorflow as tf
import numpy as np
import time
import os

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
IMAGE_CHANNELS = 3
NETWORK_DEPTH = 4

data_dir = os.getcwd() + "/fruits-360/"
train_dir = data_dir + "Training/"
validation_dir = data_dir + "Test/"

batch_size = 50
input_size = IMAGE_HEIGHT*IMAGE_WIDTH*NETWORK_DEPTH
num_classes = len(os.listdir(train_dir))
print("Total number of different fruits: {}".format(num_classes))

dropout = 0.8
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

initial_learning_rate = 0.001
final_learning_rate = 0.0001
learning_rate = initial_learning_rate

# iterations = 40000
iterations = 25000
display_interval = 50

useCkpt = False


# -------------------- Write/Read TF record logic --------------------
class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        self._sess = tf.Session()
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def write_image_data(dir_name, tfrecords_name):
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    coder = ImageCoder()
    image_count = 0
    index = -1
    classes_dict = {}

    for folder_name in os.listdir(dir_name):
        class_path = dir_name + '/' + folder_name + '/'
        index += 1
        classes_dict[index] = folder_name
        for image_name in os.listdir(class_path):
            image_path = class_path + image_name
            image_count += 1
            with tf.gfile.FastGFile(image_path, 'rb') as f:
                image_data = f.read()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            'image_raw': tf.train.Feature(bytes_list=
                                                          tf.train.BytesList(value=
                                                                             [tf.compat.as_bytes(image_data)]))
                        }
                    )
                )
                writer.write(example.SerializeToString())
    writer.close()
    print(classes_dict)
    return image_count, classes_dict


def read_image_data(filename):
    file_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    label = tf.cast(features['label'], tf.int32)
    return image, label


train_images_count, fruit_labels = write_image_data(train_dir, 'train.tfrecord')
test_images_count, _ = write_image_data(validation_dir, 'test.tfrecord')


# ------------------------- Network Structure ------------------------
def conv2d(op_name, x, W, b, strides=1):
    return tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(x, W,
                         strides=[1, strides, strides, 1],
                         padding='SAME', name=op_name), b))


def maxpool2d(op_name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding='SAME',
                          name=op_name)


def norm1(op_name, x):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name=op_name)


initializer = tf.contrib.layers.xavier_initializer()


def _variable_with_weight_decay(name, shape, initializer=initializer):
    return tf.Variable(initializer(shape), name=name)


# perform data augmentation on images
# add random hue and saturation
# randomly flip the image vertically and horizontally
# converts the image from RGB to HSV and
# adds a 4th channel to the HSV ones that contains the image in gray scale
def adjust_image_for_train(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_hue(image, 0.02)
    image = tf.image.random_saturation(image, 0.9, 1.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.flip_up_down(image)

    hsv_image = tf.image.rgb_to_hsv(image)
    gray_image = tf.image.rgb_to_grayscale(image)

    return tf.concat([hsv_image, gray_image], 2)


def adjust_image_for_test(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    gray_image = tf.image.rgb_to_grayscale(image)
    image = tf.image.rgb_to_hsv(image)
    return tf.concat([image, gray_image], 2)


def train_inputs(filename, batch_size):
    image, label = read_image_data(filename)
    image = adjust_image_for_train(image)
    return tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                  capacity=30000 + batch_size,
                                  min_after_dequeue=5000)


def test_inputs(filename, batch_size):
    image, label = read_image_data(filename)
    image = adjust_image_for_train(image)
    return tf.train.batch([image, label], batch_size=batch_size,
                          capacity=test_images_count + batch_size)


def conv_net(X, weights, biases, dropout):
    X = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, NETWORK_DEPTH])

    conv1 = conv2d('conv1', X, weights['conv_weight1'], biases['conv_bias1'])
    conv1 = maxpool2d('max_pool1', conv1, k=2)

    conv2 = conv2d('conv2', conv1, weights['conv_weight2'], biases['conv_bias2'])
    conv2 = maxpool2d('max_pool2', conv2, k=2)

    conv3 = conv2d('conv3', conv2, weights['conv_weight3'], biases['conv_bias3'])
    conv3 = maxpool2d('max_pool3', conv3, k=2)

    conv4 = conv2d('conv4', conv3, weights['conv_weight4'], biases['conv_bias4'])
    conv4 = maxpool2d('max_pool4', conv4, k=2)

    fc1 = tf.reshape(conv4, shape=[-1, weights['fc1_weight1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['fc1_weight1']), biases['fc1_bias1']))
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['fc1_weight2']), biases['fc1_bias2']))
    fc2 = tf.nn.dropout(fc2, dropout)

    return tf.add(tf.matmul(fc2, weights['out_weight']), biases['out_bias'], name='softmax')


def update_learning_rate(acc, learn_rate):
    return learn_rate - acc * learn_rate * 0.9


weights = {
    'conv_weight1': _variable_with_weight_decay('conv_weight1', [5, 5, 4, 16]),
    'conv_weight2': _variable_with_weight_decay('conv_weight2', [5, 5, 16, 32]),
    'conv_weight3': _variable_with_weight_decay('conv_weight3', [5, 5, 32, 64]),
    'conv_weight4': _variable_with_weight_decay('conv_weight4', [5, 5, 64, 128]),
    'fc1_weight1': _variable_with_weight_decay('fc1_weight1', [7*7*128, 1024]),
    'fc1_weight2': _variable_with_weight_decay('fc1_weight2', [1024, 256]),
    'out_weight': _variable_with_weight_decay('out_weight', [256, num_classes])
}
biases = {
    'conv_bias1': tf.Variable(tf.zeros([16])),
    'conv_bias2': tf.Variable(tf.zeros([32])),
    'conv_bias3': tf.Variable(tf.zeros([64])),
    'conv_bias4': tf.Variable(tf.zeros([128])),
    'fc1_bias1': tf.Variable(tf.zeros([1024])),
    'fc1_bias2': tf.Variable(tf.zeros([256])),
    'out_bias': tf.Variable(tf.zeros([num_classes]))
}


# placeholder for input layer
X = tf.placeholder(tf.float32, shape=[None, input_size], name='X')
# placeholder for actual labels
Y = tf.placeholder(tf.int64, shape=[batch_size], name='Y')

logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)


loss_operation = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=Y))

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss_operation)

correct_prediction = tf.equal(tf.argmax(prediction, 1), Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()


def train_model():
    time1 = time.time()

    for i in range(1, iterations + 1):
        with tf.Graph().as_default():
            batch_x, batch_y = sess.run([images, labels])
            batch_x = np.reshape(batch_x, [batch_size, input_size])

            sess.run(train_op, feed_dict={X: batch_x,
                                          Y: batch_y,
                                          keep_prob: dropout})

            if i % display_interval == 0 or i == 1:
                loss, acc = sess.run([loss_operation, accuracy], feed_dict={X: batch_x,
                                                                            Y: batch_y,
                                                                            keep_prob: 1})
                learning_rate = update_learning_rate(acc, learn_rate=initial_learning_rate)
                # save the weights and metadata for graph
                saver.save(sess, './model.ckpt')
                tf.train.write_graph(sess.graph_def, '.', 'graph.pbtext')
                time2 = time.time()
                print("time: %.4f step: %d loss: %.4f accuracy: %.4f" % (time2 - time1, i, loss, acc))
                time1 = time.time()


def test_model():
    images_left_to_process = test_images_count
    correct = 0
    while images_left_to_process > 0:
        batch_x, batch_y = sess.run([images, labels])
        batch_x = np.reshape(batch_x, [batch_size, input_size])

        results = sess.run(correct_prediction, feed_dict={X: batch_x,
                                                          Y: batch_y,
                                                          keep_prob: 1})
        images_left_to_process -= batch_size
        correct = correct + np.sum(results)
        print("Predicted %d out of %d; partial accuracy %.4f" % (correct,
                                                                 test_images_count - images_left_to_process,
                                                                 correct / (test_images_count - images_left_to_process)))
        print("Final accuracy is %.4f" % (correct / test_images_count))

# ------------------------------------------------------------


saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    train_file = "train.tfrecord"
    test_file = "test.tfrecord"
    images, labels = train_inputs(train_file, batch_size)
    train_coord = tf.train.Coordinator()
    train_threads = tf.train.start_queue_runners(sess=sess, coord=train_coord)

    if useCkpt:
        ckpt = tf.train.get_checkpoint_state('.')
        saver.restore(sess, ckpt.model_checkpoint_path)

    train_model()

    images, labels = test_inputs(test_file, batch_size)
    test_coord = tf.train.Coordinator()
    test_threads = tf.train.start_queue_runners(sess=sess, coord=test_coord)

    test_model()

    train_coord.request_stop()
    train_coord.join(train_threads)
    test_coord.request_stop()
    test_coord.join(test_threads)
    sess.close()
