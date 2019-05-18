import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy.sparse.linalg import spsolve
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # set to -1 to enable CPU, set to 0 to enable GPU

def conjgrad_tf(A_weights, b, x, n):
    result = {}
    #r = b - A.dot(x)       # python method
    padded_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        # reshape
    A_dotx_conv = tf.nn.conv2d(input=padded_x, filter=A_weights, strides=[1, 1, 1, 1], padding='VALID')

    A_dotx_conv = tf.reshape(A_dotx_conv, (110,1))
    r = b - A_dotx_conv
    p = r
    #rsold = np.dot(r.T, r)     # python method
    rsold = tf.matmul(tf.transpose(r), r)
    for i in range(n):
        #Ap = A.dot(p)      # python method
        padded_p = tf.pad(tf.reshape(p, (1, 10, 11, 1)), [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        Ap_c = tf.nn.conv2d(input=padded_p, filter=A_weights, strides=[1, 1, 1, 1], padding='VALID')
        Ap = tf.reshape(Ap_c, (110,1))
        # Ap = Ap_c[0, 0, :, :]
        #alpha = rsold / np.dot(p.T, Ap)        # python method
        alpha = rsold / tf.matmul(tf.transpose(p), Ap)
        x = tf.reshape(x, (110, 1))
        x = x + alpha * p
        r = r - alpha * Ap
        #rsnew = np.dot(r.T, r)     # python method
        rsnew = tf.matmul(tf.transpose(r), r)
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        #print('Itr:', i)
    result['final'] = x
    return result
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

if __name__ == '__main__':
    tol = 1e-5  # Tolerance: Decrease for grater accuracy
    conductivity = tf.Variable(1., tf.float32)
    # Filter
    filter = np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    A_weights = np.reshape(filter, (3, 3, 1, 1))* conductivity
    n = 36
    b = tf.placeholder(tf.float32, shape=(110, 1), name="b")
    x_input_pl = tf.placeholder(tf.float32, shape=(110, 1), name="x")
    x = tf.reshape(x_input_pl, (1, 10, 11, 1))
    CGpy_result = conjgrad_tf(A_weights, b, x, n)
    x = tf.reshape(x, (110, 1))
    # optimizer
    CGpy_result['loss'] = loss = tf.reduce_mean(tf.abs(CGpy_result['final'] - x))
    lr = 1
    learning_rate = tf.Variable(lr)  # learning rate for optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)  #
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads)

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    # 10 x 10 Element Data
    data1 = sio.loadmat('./data/10x10/K_forceboundary_elements10x10.mat')
    data2 = sio.loadmat('./data/10x10/f_forceboundary_elements10x10.mat')
    data3 = sio.loadmat('./data/10x10/x0_elements10x10.mat')
    A10 = data1['K_forceboundary_elements10x10']
    b10 = data2['f_forceboundary_elements10x10']
    x = spsolve(A10, b10)
    x = x.reshape(110, 1)
    b10 = np.float32(b10)
    x = np.float32(x)
    test_loss_hist = []
    train_loss_hist = []
    k_value_hist = []
    for itr in range(500):
        for i in range(1):
            x_input = x
            b_input = b10
            feed_dict_train = {b: b_input, x_input_pl: x_input}
            _, loss_value, k_value = sess.run([train_op, loss, conductivity], feed_dict_train)

            print("iter:{}  train_cost: {}  k_value: {}".format(itr, np.mean(loss_value), k_value))

    print('done')
