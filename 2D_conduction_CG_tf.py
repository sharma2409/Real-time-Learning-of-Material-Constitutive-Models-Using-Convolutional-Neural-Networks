import numpy as np
import scipy.io as sio
from timeit import default_timer as timer
import tensorflow as tf
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # set to -1 to enable CPU, set to 0 to enable GPU

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def conjgrad_tf(A_tf, b, x, n):
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
    return x


if __name__ == '__main__':


    # 10 x 10 Element Data
    data1 = sio.loadmat('./data/10x10/K_forceboundary_elements10x10.mat')
    data2 = sio.loadmat('./data/10x10/f_forceboundary_elements10x10.mat')
    data3 = sio.loadmat('./data/10x10/x0_elements10x10.mat')
    A10 = data1['K_forceboundary_elements10x10']
    b10 = data2['f_forceboundary_elements10x10']
    x10 = data3['x0_elements10x10']
    A_tensor = convert_sparse_matrix_to_sparse_tensor(A10)
    A_tf10 = tf.cast(A_tensor, tf.float32)
    b_tf10 = tf.convert_to_tensor(b10, dtype=tf.float32)
    x0_tf10 = tf.convert_to_tensor(x10, dtype=tf.float32)


    # 100 x 100 Element Data
    data4 = sio.loadmat('./data/100x100/K_forceboundary_elements100x100.mat')
    data5 = sio.loadmat('./data/100x100/f_forceboundary_elements100x100.mat')
    data6 = sio.loadmat('./data/100x100/x0_elements100x100.mat')
    A100 = data4['K_forceboundary_elements100x100']
    b100 = data5['f_forceboundary_elements100x100']
    x100 = data6['x0_elements100x100']
    A_tensor = convert_sparse_matrix_to_sparse_tensor(A100)
    A_tf100 = tf.cast(A_tensor, tf.float32)
    b_tf100 = tf.convert_to_tensor(b100, dtype=tf.float32)
    x0_tf100 = tf.convert_to_tensor(x100, dtype=tf.float32)

    # 1000 x 1000 Element Data
    data7 = sio.loadmat('./data/1000x1000/K_forceboundary_elements1000x1000.mat')
    data8 = sio.loadmat('./data/1000x1000/f_forceboundary_elements1000x1000.mat')
    data9 = sio.loadmat('./data/1000x1000/x0_elements1000x1000.mat')
    A1000 = data7['K_forceboundary_elements1000x1000']
    b1000 = data8['f_forceboundary_elements1000x1000']
    x1000 = data9['x0_elements1000x1000']
    A_tensor = convert_sparse_matrix_to_sparse_tensor(A1000)
    A_tf1000 = tf.cast(A_tensor, tf.float32)
    b_tf1000 = tf.convert_to_tensor(b1000, dtype=tf.float32)
    x0_tf1000 = tf.convert_to_tensor(x1000, dtype=tf.float32)


    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    # 10 x 10 Elements
    n10 = 36    # Based on # of python iterations
    start_tf10 = timer()
    x_result_tf10 = conjgrad_tf(A_tf10, b_tf10, x0_tf10, n10)
    end_tf10 = timer()
    print('Tensorflow solved for 10 element case in ',  end_tf10 - start_tf10, ' Seconds.')

    # 100 x 100 Elements
    n100 = 313  # Based on # of python iterations
    start_tf100 = timer()
    x_result_tf100 = conjgrad_tf(A_tf100, b_tf100, x0_tf100, n100)
    end_tf100 = timer()
    print('Tensorflow solved for 100 element case in ', end_tf100 - start_tf100, ' Seconds.')

    # 1000 x 1000 Elements
    n1000 =  2818 # Based on # of python iterations
    start_tf1000 = timer()
    x_result_tf1000 = conjgrad_tf(A_tf1000, b_tf1000, x0_tf1000, n1000)
    end_tf1000 = timer()
    print('Tensorflow solved for 1000 element case in ', end_tf1000 - start_tf1000, ' Seconds.')