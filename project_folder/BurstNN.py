import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            y = f(x)
        dy_dx = tape2.gradient(y, x)
    d2y_dx2 = tape1.gradient(dy_dx,x)
    return tf.square(d2y_dx2 + x*y)

def custom_loss(x):
    def loss(y_true, y_pred):
        differential_loss_term = tf.math.reduce_sum(tf.map_fn(differential_loss, x_train_tf))
        boundary_loss_term = 0 #tf.square(f(np.asarray([0]))[0][0])
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

# Define Neural network
i = 10
neural_net = build_model(i)

# Define optimization routine
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07)

# Training (must include initial condition point)
x_train_tf = tf.reshape(tf.convert_to_tensor(np.linspace(0,1,n)), (n,1))
neural_net.compile(loss = custom_loss(x_train_tf), optimizer = optimizer, experimental_run_tf_function = False)
epochs = 10000
fit = neural_net.fit(x = x_train_tf, y = x_train_tf, batch_siz = n, epochs = epochs)

neural_net.trainable_variables
#plotting 
x_predict = tf.convert_to_tensor(np.linspace(0,1, 1000))
y_predict = neural_net.predict(x_predict)
y_predict = np.reshape(y_predict, 1000)
#y_true = np.exp(-x_predict)
y_true = np.exp(-x_predict)*np.sin(x_predict)

fig = plt.figure()
plt.plot(x_predict, y_predict, ".")
plt.plot(x_predict, y_true)
