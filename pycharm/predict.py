import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import math
import pickle
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys


#################處理資料#############
def get_data_set(name="train", cifar=10):
    x = None
    y = None
    l = None
    maybe_download_and_extract()

    folder_name = "cifar_10" if cifar == 10 else "cifar_100"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    return x, dense_to_one_hot(y), l


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    cifar_100_directory = main_directory+"cifar_100/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_100 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.rename(main_directory+"./cifar-100-python", cifar_100_directory)
        os.remove(zip_cifar_10)
        os.remove(zip_cifar_100)

##########卷積類神經建立模型####################

def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    _RESHAPE_SIZE = 4*4*128

    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    def _variable_with_weight_decay(name, shape, stddev, wd):
        """
        參數:
          name: 變數名稱
          shape: 整數串列
          stddev: 高斯標準偏差
          wd: add L2Loss weight decay multiplied by this float.
        回傳:
          變數張量
        """
        dtype = tf.float32
        var = _variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    def put_kernels_on_grid(kernel, pad=1):

        '''卷積類神經.  特徵圖形.
         參數:
          kernel:            維度的張量 [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):  網格的維度. Require: NumKernels == grid_Y * grid_X

        回傳:
          維度的張量 [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
        '''

        # 得到網格的維度. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(math.sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # 填充 X 和 Y
        x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X 和 Y 大小
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # 放置 NumKernels 到第一維度
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # 在y軸組織網格
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # 切換X和Y軸
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # 在X軸組織網格
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # tf.image_summary  [batch_size, height, width, channels]
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        return x7

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    with tf.variable_scope('Visualization'):
        grid = put_kernels_on_grid(kernel)
        tf.summary.image('conv1/filters', grid, max_outputs=1)
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fully_connected1') as scope:
        reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('output') as scope:
        weights = _variable_with_weight_decay('weights', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [_NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(softmax_linear, dimension=1)

    return x, y, softmax_linear, global_step, y_pred_cls

#######################測試預測和觀看結果############################

test_x, test_y, test_l = get_data_set("test", cifar=10)
x, y, output, global_step, y_pred_cls = model()

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10
_SAVE_PATH = "./tensorboard/cifar-10/"

saver = tf.train.Saver()
sess = tf.Session()

try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


i = 0
predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
while i < len(test_x):
    j = min(i + _BATCH_SIZE, len(test_x))
    batch_xs = test_x[i:j, :]
    batch_ys = test_y[i:j, :]
    predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
    i = j

correct = (np.argmax(test_y, axis=1) == predicted_class)
acc = correct.mean()*100
correct_numbers = correct.sum()
print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
for i in range(_CLASS_SIZE):
    class_name = "({}) {}".format(i, test_l[i])
    print(cm[i, :], class_name)
class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
print("".join(class_numbers))


sess.close()
