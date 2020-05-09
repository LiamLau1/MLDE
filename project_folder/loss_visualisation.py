import os
import sys
sys.path.append("../GradVis/toolbox")
import tensorflow as tf
import itertools
import numpy as np
import matplotlib.pyplot as plt
import Visualization as vis
import nn_model
import trajectory_plots as tplot

# Set tensorflow backend accuracy
tf.keras.backend.set_floatx('float64')
n = 100

# Approximate function
@tf.function
def f(x):
    f = neural_net(x)
    return f

# Custom Cost function
def differential_loss(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = f(x)
    dy_dx = tape.gradient(y, x)
    return tf.square(dy_dx + y - tf.math.exp(-x) * tf.math.cos(x))

def custom_loss(x):
    def loss(y_true, y_pred):
        #differential_loss_term = tf.math.reduce_sum(tf.map_fn(differential_loss, x))
        differential_loss_term = tf.math.reduce_sum(differential_loss(x))
        boundary_loss_term = tf.square(f(np.asarray([0]))[0][0])
        return differential_loss_term/n + boundary_loss_term
    return loss

# Build Neural Network
def build_model(i):
    # Build Neural Network with i input nodes and j hidden nodes 
    input_tensor = tf.keras.layers.Input(shape=(1,))
    hidden_layer_1 = tf.keras.layers.Dense(i, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = tf.nn.sigmoid)(input_tensor)
    output_layer = tf.keras.layers.Dense(1, kernel_initializer= 'GlorotNormal', bias_initializer = 'zeros', activation = tf.identity)(hidden_layer_1)
    model = tf.keras.Model(inputs = input_tensor, outputs = output_layer)

    return model

# Function that copies neural network model and weights and returns copied model and the copied 1d array of weights which we will base around in our altering
def copy_model(model, i):
    # Copy architecture 
    model_copy = build_model(i)
    #copies model training parameters into model_copy
    for a,b in zip(model_copy.trainable_variables, model.trainable_variables):
        a.assign(b)
    copied_weight_array =  np.hstack(list(itertools.chain.from_iterable(model_copy.get_weights())))
    return model_copy, copied_weight_array

# Function which alters weights of copy  
def alter_weights(model_copy, new_theta):
    # new_theta should be 1d array
    old_params = model_copy.get_weights()
    len_old_params = len(old_params)
    arrayofshapes = []
    for k in range(len_old_params):
        arrayofshapes.append(old_params[k].shape)
    # Get array of cumulative sum for number of elements in the copy_model parameter list in the right splittings
    num_elements = np.cumsum([np.prod(x) for x in arrayofshapes])
    splittings = num_elements[:-1] # we exclude the last array element as splitting doesn't need to know the end
    # split new_theta in the right splittings and reshape them 
    new_params = [np.split(new_theta, splittings)[p].reshape(arrayofshapes[p]) for p in range(len_old_params)] # We split the new parameter 1D array into the correct lengths then reshape it into the correct shapes and then put it into a list
    # We now replace copy_model trainable parameters with these new parameters
    for a,b in zip(model_copy.trainable_variables, new_params):
        a.assign(b)

# Cost surface visualization 
## Model dependent differential loss
def model_dependent_differential_loss(model,x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    dy_dx = tape.gradient(y, x)
    return tf.square(dy_dx + y - tf.math.exp(-x) * tf.math.cos(x))

# Cost value as function of parameters with fixed point as minimum and fixed x inputs
def loss_value(model):
    #differential_loss_term = tf.math.reduce_sum(tf.map_fn(differential_loss, x_train_tf))
    differential_loss_term = tf.math.reduce_sum(model_dependent_differential_loss(model, x_train_tf))
    boundary_loss_term = tf.square(f(np.asarray([0]))[0][0])
    return differential_loss_term/n + boundary_loss_term

# Give two vectors of same length as theta_star, add to theta_star with some parameter scaling
def loss_surface(model_copy, theta_star, u,v, uparameter, vparameter):
    # reset model_copy weights as theta_star (the anchor point)
    alter_weights(model_copy, theta_star)
    # u and v are 1d arrays and must be same size as theta_star, normalize and scale u and v by respective parameters
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
    theta_step = theta_star + uparameter*u + vparameter*v
    alter_weights(model_copy, theta_step)
    valueofloss = loss_value(model_copy).numpy()
    alter_weights(model_copy, theta_star)
    return valueofloss

# Define Neural network
i = 10
neural_net = build_model(i)

# Define optimization routine
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07)

# Training (must include initial condition point)
x_train_tf = tf.reshape(tf.convert_to_tensor(np.linspace(0,1,n)), (n,1))
neural_net.compile(loss = custom_loss(x_train_tf), optimizer = optimizer, experimental_run_tf_function = False)
epochs = 10000
fit = neural_net.fit(x = x_train_tf, y = x_train_tf, batch_size = n, epochs = epochs)
# Copy fitted neural net architecture and fitted weights
copy_fitted_model, theta_star = copy_model(neural_net, i)  


#plotting 
x_predict = tf.convert_to_tensor(np.linspace(0,1, 1000))
y_predict = f(x_predict)
y_predict = np.reshape(y_predict, 1000)
y_predict2 = copy_fitted_model(x_predict)
y_predict2 = np.reshape(y_predict2, 1000)
#y_true = np.exp(-x_predict)
y_true = np.exp(-x_predict)*np.sin(x_predict)

fig = plt.figure()
plt.plot(x_predict, y_predict, ".")
plt.plot(x_predict, y_true)
plt.plot(x_predict, y_predict2)

# Plotting loss surface for two vectors
## Define two vectors with same length as theta_star
#u = np.zeros(len(theta_star))
#v = np.zeros(len(theta_star))
uparam = np.linspace(-10,10,1000)
vparam = np.linspace(-10,10,1000)
uaxis, vaxis = np.meshgrid(uparam,vparam)
zaxis = np.zeros((1000,1000))

#run everytime to test
np.random.seed(5)
utest = np.random.randn(31)
vtest = np.random.randn(31)
loss_surface(copy_fitted_model, theta_star, utest, vtest, 0, 0)
loss_value(neural_net)
for i in range(10):
    for j in range(10):
        zaxis[i,j] = loss_surface(copy_fitted_model, theta_star, utest, vtest, uaxis[i,j], vaxis[i,j])

plt.contour(uaxis, vaxis,zaxis, levels = 100)
plt.show()
ax = plt.axes(projection='3d')
ax.plot_surface(uaxis, vaxis, zaxis, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface');
