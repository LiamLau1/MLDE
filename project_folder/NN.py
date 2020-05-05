import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Build architecture, working model with one hidden dense layer with leaky_relu activation function
tf.keras.backend.set_floatx('float64')
density = 100
input_layer = tf.keras.layers.Input(shape=([1])) #creates symbolic tensor for input
## Activation function for hidden layers
#activation_function = tf.nn.leaky_relu
activation_function = tf.nn.tanh
#activation_function = tf.nn.sigmoid
#activation_function = tf.nn.elu
#activation_function = tf.nn.swish
hidden_layer_1 = tf.keras.layers.Dense(density, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = activation_function)(input_layer)
hidden_layer_2 = tf.keras.layers.Dense(100, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = activation_function)(hidden_layer_1)
hidden_layer_3 = tf.keras.layers.Dense(100, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = activation_function)(hidden_layer_2)
hidden_layer_4 = tf.keras.layers.Dense(100, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = activation_function)(hidden_layer_3)
output_layer = tf.keras.layers.Dense(1, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = activation_function)(hidden_layer_4)
model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

model.summary()
tf.keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes = True)

# Define domain
x_train = np.linspace(0,2,density)
x_train_tf = tf.convert_to_tensor(x_train)
#x_train_tf = tf.reshape(x_train_tf, [100,1])


# Define custom cost function with automatic differentiation for problem 2
def loss(model,x):
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(x_train_tf)
        y = model(x_train_tf)
    dy_dx = tape.gradient(y,x_train_tf)
    return (tf.reduce_mean((dy_dx + y - tf.math.exp(-x_train_tf) * tf.math.cos(x_train_tf))**2) + (y[0])**2)
    #return (tf.reduce_mean((dy_dx + y)**2) + (y[0] - 1)**2)

# Define gradients to optimize model
def grad(model, inputs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Mean absolute error
y_label = np.exp(-x_train_tf)*np.sin(x_train_tf)
def mae(model, inputs, label):
    y_pred = model(inputs)
    return tf.reduce_mean(tf.math.abs((y_pred - label)))


# Set Optimizer
optimizer = tf.keras.optimizers.Adam()

# Train NN
## Keep results for plotting
train_loss_results = []
train_mae_results = []

num_epochs = 2001

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()

    #Optimize model
    loss_value, grads = grad(model, x_train_tf)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
     # Track progress
    epoch_loss_avg.update_state(loss_value) 
    epoch_mae = mae(model, x_train_tf ,y_label)
    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_mae_results.append(epoch_mae)
    if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.6f},  MAE: {:.4}".format(epoch,epoch_loss_avg.result(), epoch_mae))


x_predict = np.linspace(0,2, density * 100)
y_predict = model.predict(x_predict)
y_predict = np.reshape(y_predict,(density*100,))
y_true = np.exp(-x_predict)*np.sin(x_predict)

fig = plt.figure()
plt.plot(x_predict,y_predict, ".")
plt.plot(x_predict, y_true)
#plt.plot(x_train_tf, np.exp(-x_train_tf))

fig = plt.figure()
plt.plot(range(num_epochs),train_loss_results)

fig = plt.figure()
plt.plot(x_predict, (y_predict - y_true), '.')
