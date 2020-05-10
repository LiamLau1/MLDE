import os
import sys
sys.path.append("../GradVisV2/toolbox")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import time
import colorama
from matplotlib import animation
import Visualization as vis
import nn_model
import trajectory_plots as tplot


#Random seed initialization
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)



class ODEsolver():
    
    def __init__(self, x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save):
        """
        x : training domain (ex: x = np.linspace(0, 1, 100))
        initial_condition : initial condition including x0 and y0 (ex: initial_condition = (x0 = 0, y0 = 1))
        architecture : number of nodes in hidden layers (ex: architecture = [10, 10])
        initializer : weight initializer (ex: 'GlorotNormal')
        activation : activation function (ex: tf.nn.sigmoid)
        optimizer : minimization optimizer including parameters (ex: tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07))
        prediciton_save : bool to save predicitons at each epoch during training (ex: prediction_save = False)
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
        if prediction_save:
            self.predictions = []
        
        
        
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
    
    def train(self):
        """
        neural_net : The built neural network returned by self.neural_net_model
        Trains the model according to x.
        """
        x = self.x
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, (self.n, 1))
        neural_net = self.neural_net
        
        if prediction_save:
            predictions = self.predictions

            #Define custom callback for predictions during training
            class PredictionCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    y_predict = neural_net.predict(x)
                    predictions.append(y_predict)
                    print('Prediction saved at epoch: {}'.format(epoch))

            start_time = time.time()
            history = neural_net.fit(x = x, y = x, batch_size = self.n, epochs = self.epochs, callbacks = [PredictionCallback()])
            print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
            print(f"{self.RESET}")
            predictions = tf.reshape(predictions, (self.epochs, self.n))
            
        else:
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
        relative_error = tf.keras.losses.mean_absolute_error(y_exact, y_predict) 
        return relative_error


    def MAE(self, y_predict, y_exact):
        """
        y_predict : array of predicted solution
        y_exact : array of exact solution
        Returns the mean absolute error of the neural network solution
        given the exact solution.
        """
        if len(y_exact) != len(y_predict):
            raise Exception("y_predict and y_exact do not have the same shape.")
        mae = tf.keras.losses.MAE(y_exact, y_predict)
        return mae
    
    
    def get_predictions(self):
        """
        Returns the neural net predictions at each epoch 
        """
        if not prediction_save:
            raise Exception("The predictions have not been saved.")
        else:
            return self.predictions
        
        
    def training_animation(self, y_exact, y_predict, epoch, loss):
        """
        Plot the training animation including the exact solution, 
        the neural network solution and the cost function as functions of epochs.
        This function needs the model to be trained and requires the outputs of get_loss.
        """
        if not prediction_save:
            raise Exception("The predictions have not been saved.")
        fig, ax = plt.subplots()
        ax1 = plt.axes()
        ax2 = fig.add_axes([0.58, 0.2, 0.3, 0.2])

        frames = []

        x = self.x
        predictions = self.predictions

        x_loss = epoch
        y_loss = loss

        ax1.plot(x, y_exact, "C1", label = "Exact solution")
        ax1.legend(loc = 'upper left')
        ax1.set_xlim(min(x), max(x))
        ax1.set_ylim(min(y_predict) - 0.1, max(y_predict) + 0.1)
        ax1.set_xlabel("$x$", fontsize = 15)
        ax1.set_ylabel("$\hat{f}$", fontsize = 15)

        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.semilogy(x_loss, y_loss, color = "w", linewidth = 1.0)

        x_loss_points = []
        y_loss_points = []

        for i in range(self.epochs):
            y = predictions[i]
            frame1, = ax1.plot(x, y, ".", color = "C0")
            x_loss_points.append(x_loss[i])
            y_loss_points.append(y_loss[i])
            frame2, = ax2.semilogy(x_loss_points, y_loss_points, color = "C0")
            frames.append([frame1, frame2])

        ani = animation.ArtistAnimation(fig, frames, interval = 20, blit = True)
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('./data/animation.mp4', writer=writer)
        #plt.show()
        
    
        



if __name__ == "__main__":

        #--------------------------------------------------------------------
        #-----------------PARAMETER-INITIALIZATION---------------------------
        #--------------------------------------------------------------------

        #Training domain
        x = np.linspace(0, 1, 100)
        #Initial conditions
        initial_condition = (0, 0)
        #Number of epochs
        epochs = 2000
        #Structure of the neural net (only hidden layers)
        architecture = [10]
        #Initializer used
        initializer = 'GlorotNormal'
        #Activation function used
        activation = tf.nn.sigmoid
        #Optimizer used
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07)
        #Save predictions at each epoch
        prediction_save = False

        #--------------------------------------------------------------------
        #------------------MODEL-DEFINITION-AND-TRAINING---------------------
        #--------------------------------------------------------------------

        #Class definition
        solver = ODEsolver(x, initial_condition, epochs, architecture, initializer, activation, optimizer, prediction_save)
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
        #plt.plot(x_predict, y_exact, label = "Exact solution")
        #plt.plot(x_predict, y_predict, ".", label = "Neural network solution")
        #plt.legend()
        #plt.savefig("./data/prediction.pdf")

        #Plot the relative error
        relative_error = solver.relative_error(y_predict, y_exact)
        #plt.semilogy(x_predict, relative_error)
        #plt.show()

        #--------------------------------------------------------------------
        #------------------TRAINING-ANIMATION--------------------------------
        #--------------------------------------------------------------------

        #solver.training_animation(y_exact, y_predict, epoch, loss)

        #--------------------------------------------------------------------
        #------------------LOSS-SURFACE--------------------------------
        #--------------------------------------------------------------------
        solver.neural_net.save_weights('./data/minimum_0')
        nnmodel = nn_model.Tensorflow_NNModel(solver.neural_net, solver.neural_net.loss, solver.x, './data/minimum_0')
        # proz uses to scale from [-5,5] domain for loss surface plotting, find a spacing n such that it gets as close to 0 as possible
        #filenames needs to be an array of files
        vis.visualize(nnmodel,solver.neural_net.loss,solver.x, ['./data/minimum_0'], 200, './data/example',random_dir = True, proz = 0.5, verbose=True)
        tplot.plot_loss_2D('./data/example.npz','./data/plot2d',is_log=False)
        tplot.plot_loss_3D('./data/example.npz','./data/plot3d',is_log=False, degrees = 120)
        outs = np.load('./data/example.npz', allow_pickle=True)
        outs = outs["a"]
        tplot.plot3D(outs[0][0],outs[0][1],outs[0][2])
        np.min(outs[0][2])

