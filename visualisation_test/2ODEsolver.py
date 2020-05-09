import os
import sys
sys.path.append("../GradVisV2/toolbox")
import tensorflow as tf
import numpy as np
#import Tkinter as tk
import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import time
import colorama
import Visualization as vis
import nn_model
import trajectory_plots as tplot


#Random seed initialization
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)



class ODEsolver():
    
    def __init__(self, x, initial_condition, epochs, architecture, initializer, activation, optimizer):
        """
        x : training domain (ex: x = np.linspace(0, 1, 100))
        initial_condition : initial condition including x0 and y0 (ex: initial_condition = (x0 = 0, y0 = 1))
        architecture : number of nodes in hidden layers (ex: architecture = [10, 10])
        initializer : weight initializer (ex: 'GlorotNormal')
        activation : activation function (ex: tf.nn.sigmoid)
        optimizer : minimization optimizer including parameters (ex: tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07))
        """
        colorama.init()
        self.GREEN = colorama.Fore.GREEN
        self.RESET = colorama.Fore.RESET
        tf.keras.backend.set_floatx('float64')
        self.x = x
        self.initial_condition = initial_condition
        self.n = len(self.x)
        self.epochs = epochs
        self.architecture = architecture
        self.initializer = initializer
        self.activation = activation
        self.optimizer = optimizer
        self.neural_net = self.build_model()#self.neural_net_model(show = True)
        self.neural_net.summary()
        
        #Compile the model
        x = self.x
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, (self.n, 1))
        self.neural_net.compile(loss = self.custom_cost(x), optimizer = self.optimizer, experimental_run_tf_function = False)
        print("------- Model compiled -------")
        
        
        
    def build_model(self):
        """
        Builds a customized neural network model.
        """
        architecture = self.architecture
        initializer = self.initializer
        activation = self.activation
        
        nb_hidden_layers = len(architecture)
        input_tensor = tf.keras.layers.Input(shape = (1,))
        hidden_layers = []

        if nb_hidden_layers >= 1:
            hidden_layer = tf.keras.layers.Dense(architecture[0], kernel_initializer= initializer, bias_initializer='zeros',activation = activation)(input_tensor)
            hidden_layers.append(hidden_layer)
            for i in range(1, nb_hidden_layers):
                hidden_layer = tf.keras.layers.Dense(architecture[i], kernel_initializer= initializer, bias_initializer='zeros',activation = activation)(hidden_layers[i-1])
                hidden_layers.append(hidden_layer)
            output_layer = tf.keras.layers.Dense(1, kernel_initializer= initializer, bias_initializer = 'zeros', activation = tf.identity)(hidden_layers[-1])
        else:
            output_layer = tf.keras.layers.Dense(1, kernel_initializer= initializer, bias_initializer = 'zeros', activation = tf.identity)(input_tensor)
        
        model = tf.keras.Model(inputs = input_tensor, outputs = output_layer)
        return model
    
    
    @tf.function
    def NN_output(self, x):
        """
        x : must be of shape = (?, 1)
        Returns the output of the neural net
        """
        y = self.neural_net(x)
        return y
    

    def y_gradients(self, x):
        """
        Computes the gradient of y.
        """
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y = self.NN_output(x)
            dy_dx = tape2.gradient(y, x)
        d2y_dx2 = tape1.gradient(dy_dx,x)
        return y, dy_dx, d2y_dx2

    
    def differential_cost(self, x):
        """
        Defines the differential cost function for one neural network
        input.
        """
        y, dy_dx, d2y_dx2 = self.y_gradients(x)

        #----------------------------------------------
        #------------DIFFERENTIAL-EQUATION-------------
        #----------------------------------------------
        differential_equation = dy_dx + y - tf.math.exp(-x) * tf.math.cos(x)
        #----------------------------------------------
        #----------------------------------------------
        #----------------------------------------------

        return tf.square(differential_equation)


    def custom_cost(self, x):
        """
        Defines the cost function for a batch.
        """
        x0 = self.initial_condition[0]
        y0 = self.initial_condition[1]
        def loss(y_true, y_pred):
            differential_cost_term = tf.math.reduce_sum(self.differential_cost(x))
            boundary_cost_term = tf.square(self.NN_output(np.asarray([[x0]]))[0][0] - y0)
            return differential_cost_term/self.n + boundary_cost_term
        return loss

    def cost_value(self,x):
        """
        Returns cost_value for cost surface plotting
        """
        x0 = self.initial_condition[0]
        y0 = self.initial_condition[1]
        differential_cost_term = tf.math.reduce_sum(self.differential_cost(x))
        print(differential_cost_term)
        boundary_cost_term = tf.square(self.NN_output(np.asarray([[x0]]))[0][0] - y0)
        cost =  differential_cost_term/self.n + boundary_cost_term
        return cost
    
    
    def train(self):
        """
        neural_net : The built neural network returned by self.neural_net_model
        Trains the model according to x.
        """
        x = self.x
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, (self.n, 1))
        neural_net = self.neural_net
        start_time = time.time()
        history = neural_net.fit(x = x, y = x, batch_size = self.n, epochs = self.epochs)
        print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
        print(f"{self.RESET}")
        return history
    
    
    def get_loss(self, history):
        """
        history : history of the training procedure returned by self.train
        Returns epochs and loss
        """
        epochs = history.epoch
        loss = history.history["loss"]
        return epochs, loss
    
    
    def predict(self, x_predict):
        """
        x_predict : domain of prediction (ex: x_predict = np.linspace(0, 1, 100))
        """
        domain_length = len(x_predict)
        x_predict = tf.convert_to_tensor(x_predict)
        x = tf.reshape(x_predict, (domain_length, 1))
        y_predict = self.neural_net.predict(x_predict)
        return y_predict


    def relative_error(self, y_predict, y_exact):
        """
        y_predict : array of predicted solution
        y_exact : array of exact solution
        Returns the relative error of the neural network solution
        given the exact solution.
        """
        if len(y_exact) != len(y_predict):
            raise Exception("y_predict and y_exact do not have the same shape.")
        relative_error = np.abs((np.reshape(y_predict, [self.n]) - y_exact)/(y_exact))
        return relative_error
        
    
        
#global solver

if __name__ == "__main__":

    #--------------------------------------------------------------------
    #-----------------PARAMETER-INITIALIZATION---------------------------
    #--------------------------------------------------------------------

    #Training domain
    x = np.linspace(0, 1, 100)
    #Initial conditions
    initial_condition = (0, 0)
    #Number of epochs
    epochs = 10000
    #Structure of the neural net (only hidden layers)
    architecture = [10]
    #Initializer used
    initializer = 'GlorotNormal'
    #Activation function used
    activation = tf.nn.sigmoid
    #Optimizer used
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07)

    #--------------------------------------------------------------------
    #------------------MODEL-DEFINITION-AND-TRAINING---------------------
    #--------------------------------------------------------------------

    #Class definition
    # tells python that this refers to the global solver variable
    #global solver
    solver = ODEsolver(x, initial_condition, epochs, architecture, initializer, activation, optimizer)
    #Training
    history = solver.train()
    epoch, loss = solver.get_loss(history)

    #--------------------------------------------------------------------
    #------------------PREDICTION----------------------------------------
    #--------------------------------------------------------------------

    #Plot the exact and the neural net solution
    x_predict = x
    y_predict = solver.predict(x_predict)
    y_exact = np.exp(-x_predict)*np.sin(x_predict)
    plt.plot(x_predict, y_exact, label = "Exact solution")
    plt.plot(x_predict, y_predict, ".", label = "Neural network solution")
    plt.legend()
    plt.show()

    #Plot the relative error
    relative_error = solver.relative_error(y_predict, y_exact)
    plt.semilogy(x_predict, relative_error)
    plt.show()

    solver.neural_net.save_weights('./data/minimum_0')
    nnmodel = nn_model.Tensorflow_NNModel(solver.neural_net, solver.cost_value, solver.x, './data/minimum_0')
    vis.visualize(nnmodel,solver.cost_value,solver.x, './data/minimum_0', 80, './data/example',random_dir = True, proz = 0.5, verbose=True)
    tplot.plot_loss_2D('./data/example.npz','./data/plot',is_log=False)
    tplot.plot_loss_3D('./data/example.npz','./data/plot',is_log=False, degrees = 50)

#global solver
