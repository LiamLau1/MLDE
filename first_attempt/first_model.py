import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

np.random.seed(1234)

# NN classes
class NN_model:


   def __init__(self, X, layers):

       # Define self attributes
       self.xinput = tf.Variable(X, dtype = tf.float32)

       # Initialize NN

       self.layers = layers
       self.weights, self.biases = self.initialize_NN(layers)

       # tf graphs

       self.f_pred = self.net_DE(self.xinput)
    
       # Loss
       """Want f(phi, grad_phi) = 0, this squared will be our loss"""
       self.loss = tf.reduce_mean(tf.square(self.f_pred))

       # Optimizers
       self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 50000,'maxfun': 50000,'maxcor': 50,'maxls': 50,'ftol' : 1.0 * np.finfo(float).eps})

       self.optimizer_Adam = tf.train.AdamOptimizer()
       self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

   def initialize_NN(self, layers):

       weights = []
       biases = []
       num_layers = len(layers) # number of layers
       # populate the weights and biases tensors
       for l in range(0, num_layers-1):
           # return each weight matrix and bias value and append into weights and biases tensors
           w = self.xavier_init(size = [layers[l], layers[l+1]])
           # we will initialize bias vector with zeros 
           b = tf.Variable(tf.zeros([layers[l],1], dtype = tf.float32), dtype = tf.float32)
           weights.append(w)
           biases.append(b)
       # This executes
       return weights, biases


   def xavier_init(self, size):

       """ Xavier initialization is such that the initial weights aren't too small or too big. Read the README for more information. """
       n_in = size[0]
       n_out = size[1]
       std = np.sqrt(2/(n_in + n_out))
       # Use truncated normal distribution of standard deviation std, truncated means if outside 2 sigma, it is repicked.
       return tf.Variable(tf.random.truncated_normal([n_in, n_out], mean = 0, stddev = std), dtype = tf.float32)

   def neural_network(self, X_inputs, weights, biases):

       num_layers = len(weights) + 1 # weights is some array of matrix values for weights.
       # Hidden node values
       H = []
       for l in range(0, num_layers-2):
           W = weights[l]
           b = biases[l]
           if l ==0:
               print(tf.matmul(X_inputs, W))
               print(tf.add(tf.matmul(X_inputs, W),b))
               H_value = tf.tanh(tf.add(tf.matmul(X_inputs, W),b))
               print(H_value)
               H.append(H_value)
           else:
               H_value = tf.tanh(tf.add(tf.matmul(H[l], W), biases))
               H.append(H_value)

       # Output 
       W = weights[-1]
       b = biases[-1]
       # not sure if activation function needed for output node
       output = tf.add(tf.matmul(H[-1], W), b)
       
       return output

   def net_DE(self, x):

       """ Get gradients and combines them to get F hat, value of f for trial solutions. I think n of these functions are needed for a differential equation of order n """
       with tf.GradientTape(watch_accessed_variables=False) as tape:
           tape.watch(self.xinput)
           y = self.neural_network(self.xinput, self.weights, self.biases)
           dy_dx = tape.gradient(y, self.xinput)
           F = y + dy_dx

       return F

   def callback(self, loss):

       print('Loss:', loss)

   def train(self, nIter):

       """ For nIter iterations, reduce the loss function ie train the net """
       start_time = time.time()

       for iteration in range(nIter):
           self.train_op_Adam
           # Print values
           if it % 10 == 0:
               elapsed_time = time.time() - start_time
               loss_value = self.loss
               print('It: %d, Loss: %.3e, Time: %.2f' % (it, loss_value, elapsed))
               start_time = time.time()
        



if __name__ == '__main__':

    layers = [5, 10, 5]
    Xs = np.array([1,2,3,4,5])
    Xs = Xs.reshape((1,5))
    model  = NN_model(Xs, layers)


