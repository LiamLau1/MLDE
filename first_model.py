import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(1234)
tf.set_random_seed(1234)

# NN classes
class model:

    def __init__(self, layers):

        # Define self attributes

        # Initialize NN

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

       # Loss
       """Want f(phi, grad_phi) = 0, this squared will be our loss"""
       self.loss = 


       def initialize_NN(self, layers):

           weights = 
           biases = 
           num_layers = len(layers) # number of layers
           # populate the weights and biases tensors
           for l in range(0, num_layers-1):
               # return each weight matrix and bias value and append into weights and biases tensors
               w = self.xavier_init(size = [layers[l], layers[l+1]])
               # we will initialize bias vector with zeros 
               b = tf.Variable(tf.zeros([layers[l],1], dtype = tf.float32), dtype = tf.float32)
               weights.append(w)
               biases.append(b)


       def xavier_init(self, size):

           """ Xavier initialization is such that the initial weights aren't too small or too big. Read the README for more information. """
           n_in = size[0]
           n_out = size[1]
           std = np.sqrt(2/(n_in + n_out))
           # Use truncated normal distribution of standard dev std, truncated means if outside 2 sigma, it is repicked.
           return tf.Variable(tf.truncated_normal([n_in, n_out], mean = 0, stddev = std), dtype = float32)

       def neural_network(self, X_inputs, weights, biases):

           num_layers = len(weights) + 1 # weights is some array of matrix values for weights.
           # Hidden node values
           H = []
           for l in range(0, num_layers-2):
               W = weights[l]
               b = biases[l]
               if l ==0:
                   H_value = tf.tanh(tf.add(tf.matmul(X_inputs, W),biases))
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






            
