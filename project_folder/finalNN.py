import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set tensorflow backend accuracy
tf.keras.backend.set_floatx('float64')

# Approximate function
def f(x):
    f = neural_net(x)
    return f

# Custom Cost function
def custom_loss(x):
    def loss(y_true, y_pred):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = f(x)
        dy_dx = tape.gradient(y, x)
        print(f(np.asarray([0])))
        return (tf.reduce_mean(tf.square(dy_dx + y - tf.math.exp(-x) * tf.math.cos(x))) + tf.square(f(np.asarray([0])).numpy()[0][0]))
    return loss

# Build Neural Network
def build_model(i,j):
    # Build Neural Network with i input nodes and j hidden nodes 
    input_tensor = tf.keras.layers.Input(shape=([1]))
    input_layer = tf.keras.layers.Dense(i, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = tf.nn.elu)(input_tensor) # layer takes in inputs and then applies weights, biases and activation function
    hidden_layer_1 = tf.keras.layers.Dense(j, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = tf.nn.tanh)(input_layer)
    output_layer = tf.keras.layers.Dense(1, kernel_initializer= 'GlorotNormal', bias_initializer = 'zeros', activation = tf.identity)(hidden_layer_1)
    model = tf.keras.Model(inputs = input_tensor, outputs = output_layer)

    return model

# Define Neural network
i = 10
j = 10
neural_net = build_model(i,j)

neural_net(np.asarray([0])).numpy()[0][0]
f(np.asarray([0])).numpy()[0][0]

# Define optimization routine
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07)

# Training (must include initial condition point)
x_train = tf.convert_to_tensor(np.linspace(0,4,i))
neural_net.compile(loss = custom_loss(x_train), optimizer = optimizer, experimental_run_tf_function = False)
epochs = 10000
fit = neural_net.fit(x = x_train, y = x_train, epochs = epochs)

#plotting 
x_predict = tf.convert_to_tensor(np.linspace(0,4, i*100))
y_predict = neural_net.predict(x_predict)
y_predict = np.reshape(y_predict, i*100)
#y_true = np.exp(-x_predict)
y_true = np.exp(-x_predict)*np.sin(x_predict)

fig = plt.figure()
plt.plot(x_predict, y_predict, ".")
plt.plot(x_predict, y_true)
