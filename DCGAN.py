# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 20:28:31 2018

@author: kosei
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape,stddev=0.02)
    return tf.Variable(initial, name=name)
    #return tf.Variable(tf.random_normal(shape, stddev=0.02), name=name)

def bias_variable(shape, name):
    initial = tf.constant(0,shape=shape)
    return tf.Variable(initial, name=name)
    #return tf.Variable(tf.random_normal(shape, stddev=0.02), name=name)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='SAME')

def deconv2d(x,W,output_shape):
    return tf.nn.conv2d_transpose(x, W, strides=[1,2,2,1], output_shape=output_shape)

def leaky_relu(x):
    return tf.maximum(0.2*x, x)

class DCGAN:
    
    def __init__(self,
                 batch_size=100,
                 image_shape = [28,28,1],
                 dim_z = 100,
                 dim_l1 = 1024,
                 dim_l2 = 128,
                 dim_l3 = 64,
                 dim_ch = 1):
        # バッチサイズ
        self.batch_size = batch_size
        # 画像サイズ
        self.image_shape = image_shape
        # 各層のユニット数
        self.dim_z = dim_z
        self.dim_l1 = dim_l1
        self.dim_l2 = dim_l2
        self.dim_l3 = dim_l3
        self.dim_ch = dim_ch
        # パラメータ定義
        ## Generator
        self.g_W1 = weight_variable([dim_z, dim_l1],'g_W1')
        self.g_b1 = bias_variable([dim_l1],'g_b1')
        self.g_W2 = weight_variable([dim_l1, dim_l2*7*7],'g_W2')
        self.g_b2 = bias_variable([dim_l2*7*7],'g_b2')
        self.g_W3 = weight_variable([5, 5, dim_l3, dim_l2],'g_W3')
        self.g_b3 = bias_variable([dim_l3],'g_b3')
        self.g_W4 = weight_variable([5, 5, dim_ch, dim_l3],'g_W4')
        self.g_b4 = bias_variable([dim_ch],'g_b4')
        ## Discriminator
        self.d_W1 = weight_variable([5, 5, dim_ch, dim_l3],'d_W1')
        self.d_b1 = bias_variable([dim_l3],'d_b1')
        self.d_W2 = weight_variable([5, 5, dim_l3, dim_l2],'d_W2')
        self.d_b2 = bias_variable([dim_l2],'d_b2')
        self.d_W3 = weight_variable([dim_l2*7*7, dim_l1],'d_W3')
        self.d_b3 = bias_variable([dim_l1],'d_b3')
        self.d_W4 = weight_variable([dim_l1, 1],'d_W4')
        self.d_b4 = bias_variable([1],'d_b4')
    
    def build_model(self):
        # Generatorの入力ノイズ
        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        
        # 画像
        img_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        img_gen = self.generate(Z, self.batch_size)
        # 出力
        p_real = self.discriminate(img_real)
        p_gen = self.discriminate(img_gen)
        
        # コスト関数の定義
        D_cost = tf.reduce_mean(-tf.reduce_sum(tf.log(p_real) + tf.log(tf.ones(self.batch_size, tf.float32) - p_gen), reduction_indices=[1]))
        G_cost = tf.reduce_mean(-tf.reduce_sum(tf.log(p_gen), reduction_indices=[1]))
        
        return Z, img_real, D_cost, G_cost
        
    def generate(self, Z, batch_size):
        
        # 1層目
        fc1 = tf.matmul(Z, self.g_W1) + self.g_b1
        bm1, bv1 = tf.nn.moments(fc1, axes=[0])
        bn1 = leaky_relu(tf.nn.batch_normalization(fc1, bm1, bv1, None, None, 1e-5))
        
        # 2層目
        fc2 = tf.matmul(bn1, self.g_W2) + self.g_b2
        bm2, bv2 = tf.nn.moments(fc2, axes=[0])
        bn2 = leaky_relu(tf.nn.batch_normalization(fc2, bm2, bv2, None, None, 1e-5))
        
        conv1 = tf.reshape(bn2, [batch_size, 7, 7, self.dim_l2])
        
        # 3層目
        conv2 = deconv2d(conv1, self.g_W3, [batch_size, 14, 14, self.dim_l3]) + self.g_b3
        bm3, bv3 = tf.nn.moments(conv2, axes=[0, 1, 2])
        bn3 = leaky_relu(tf.nn.batch_normalization(conv2, bm3, bv3, None, None, 1e-5))
        
        # 4層目
        conv3 = deconv2d(bn3, self.g_W4, [batch_size, 28, 28, self.dim_ch]) + self.g_b4
        
        img = tf.nn.sigmoid(conv3)
        
        return img
        
    def discriminate(self, img):
        
        # 1層目
        conv1 = leaky_relu(conv2d(img, self.d_W1) + self.d_b1)
        
        # 2層目
        conv2 = leaky_relu(conv2d(conv1, self.d_W2) + self.d_b2)
        
        # 3層目
        vec = tf.reshape(conv2, [self.batch_size, 7*7*self.dim_l2])
        fc1 = leaky_relu(tf.matmul(vec, self.d_W3) + self.d_b3)
        
        # 4層目
        fc2 = tf.nn.sigmoid(tf.matmul(fc1, self.d_W4) + self.d_b4)
        
        return fc2
    
    def train(self, X):
        # モデル構築
        Z, inp_x, D_loss, G_loss = self.build_model()
        # 変数
        D_vars = [x for x in tf.trainable_variables() if 'd_' in x.name]
        G_vars = [x for x in tf.trainable_variables() if 'g_' in x.name]
        # 最適化メソッド
        optimizer_d = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(D_loss, var_list=D_vars)
        optimizer_g = tf.train.AdamOptimizer(1e-3, beta1=0.5).minimize(G_loss, var_list=G_vars)
        # セッション開始
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print('学習開始')
        start = time.time()
        for i in range(30000):
            Zs = np.random.uniform(-1,1, size=[self.batch_size, self.dim_z]).astype(np.float32)
            if(np.mod(i,2) == 0):
                index = np.random.choice(range(X.shape[0]), self.batch_size, replace=False)
                batch = X[index].reshape([-1, 28, 28, 1])
                self.sess.run(optimizer_d, feed_dict={Z:Zs, inp_x:batch})
            else:
                self.sess.run(optimizer_g, feed_dict={Z:Zs})
            
            if((i+1)%1000 == 0):
                #t_loss = self.sess.run(loss, feed_dict={inp_x:batch})
                print('step'+str(i+1)+' 経過時間: '+str(time.time() - start))
                #print('loss: ' + str(t_loss))
        print('学習終了 時間:{}'.format(time.time() - start))
        #print(self.sess.run(self.g_W1, feed_dict={Z:Zs}))
    
    def generate_sample(self, rows=4, cols=8):
        Zs = np.random.uniform(-1,1, size=[rows*cols, self.dim_z]).astype(np.float32)
        sample = self.generate(Zs, rows*cols)
        sample = self.sess.run(sample)
        for index, data in enumerate(sample):
            # 画像データはrows * colsの行列上に配置
            plt.subplot(rows, cols, index + 1)
            # 軸表示は無効
            plt.axis("off")
            # データをグレースケール画像として表示
            plt.imshow(data.reshape(28,28), cmap="gray", interpolation="nearest")
        plt.savefig('generated_sample/sample.png')
        plt.show()
        
if(__name__ == '__main__'):
    dcgan_model = DCGAN()
    dcgan_model.train(mnist.train.images)
    dcgan_model.generate_sample()
    