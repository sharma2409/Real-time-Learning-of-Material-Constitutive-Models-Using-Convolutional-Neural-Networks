import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy.sparse.linalg import spsolve


def conjgrad_tf(A_tf, b, x, n):
    result = {}
    #r = b - A.dot(x)
    r = b - tf.sparse_tensor_dense_matmul(A_tf, x, adjoint_a=False, adjoint_b=False, name=None)
    p = r
    #rsold = np.dot(r.T, r)
    rsold = tf.matmul(tf.transpose(r), r)
    for i in range(n):
        #Ap = A.dot(p)
        Ap = tf.sparse_tensor_dense_matmul(A_tf, p, adjoint_a=False, adjoint_b=False, name=None)
        #alpha = rsold / np.dot(p.T, Ap)
        alpha = rsold / tf.matmul(tf.transpose(p), Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        #rsnew = np.dot(r.T, r)
        rsnew = tf.matmul(tf.transpose(r), r)
        #print('Itr:', i)
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    result['final'] = x
    return result

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

if __name__ == '__main__':
    tol = 1e-5  # Tolerance: Decrease for grater accuracy
    conductivity = tf.Variable(1., tf.float32)
    n = 36
    A = tf.sparse_placeholder(tf.float32, shape=(110, 110))
    b = tf.placeholder(tf.float32, shape=(110, 1))
    x = tf.placeholder(tf.float32, shape=(110, 1))
    CGpy_result = conjgrad_tf(A, b, x, n)

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
    #tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    # 10 x 10 Element Data
    data1 = sio.loadmat('./data/10x10/K_forceboundary_elements10x10.mat')
    data2 = sio.loadmat('./data/10x10/f_forceboundary_elements10x10.mat')
    data3 = sio.loadmat('./data/10x10/x0_elements10x10.mat')
    A10 = data1['K_forceboundary_elements10x10']
    b10 = data2['f_forceboundary_elements10x10']
    x10 = spsolve(A10, b10)
    x_gt = x10.reshape(1, 10, 11, 1)
    b10 = b10.reshape(1, 10, 11, 1)
    test_loss_hist = []
    train_loss_hist = []
    k_value_hist = []
    for itr in range(500):
        for i in range(1):
            x_input = x_gt
            b_input = b10
            feed_dict_train = {b: b_input, x: x_input}
            _, loss_value, k_value = sess.run([train_op, loss, conductivity], feed_dict_train)

            print("iter:{}  train_cost: {}  k_value: {}".format(itr, np.mean(loss_value), k_value))

    print('done')
