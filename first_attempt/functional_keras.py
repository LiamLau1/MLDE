import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Build architecture, connect hidden layer
tf.config.experimental_run_functions_eagerly(True)
input_tensor = tf.keras.layers.Input(shape=[1])
hidden = tf.keras.layers.Dense(100, kernel_initializer= 'GlorotNormal', bias_initializer='zeros', activation = tf.nn.sigmoid)(input_tensor)
#hidden1 = tf.keras.layers.Dense(50, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = tf.nn.sigmoid)(hidden)
hidden1 = tf.keras.layers.Dense(10, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = tf.nn.sigmoid)(hidden)
output = tf.keras.layers.Dense(1, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = tf.nn.sigmoid)(hidden1)
tf.keras.initializers.GlorotNormal
model = tf.keras.Model(inputs = input_tensor, outputs = output)

model.summary()

x = tf.Variable([[0]], dtype = tf.float32)
def custom_loss_function(input_tensor):

    def loss(y_true, y_pred):

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            y = model(input_tensor)
        dy_dx = tape.gradient(y, input_tensor)
        return tf.reduce_mean((dy_dx + y)**2) + tf.math.exp((y[0] - 1) **2)
        #return 10000* tf.math.exp((y[0] - 1) **2)
    #np.exp(-0.01*np.arange(0,100, 1)
    return loss

# set optimizer
optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0)
model.compile(loss=custom_loss_function(input_tensor), optimizer=optimizer, experimental_run_tf_function = False)

x_train = np.linspace(0, 5, 100)
x_train = np.reshape(x_train, newshape = (100, 1))
#Training the neural network
epochs = 1000
history = model.fit(x = x_train, y= x_train, epochs = epochs)


#Testing prediction
x_predict = np.linspace(0, 10, 1000)
y_predict = model.predict(x_predict)
y_predict = np.reshape(y_predict, newshape = (1000,))
fig = plt.figure()
plt.plot(x_predict, y_predict, ".", label = "Prediction $y$")
plt.plot(x_predict, np.exp(-x_predict))

plt.semilogy(x_predict, y_predict, label = "Prediction")
plt.semilogy(x_predict, np.exp(-x_predict), label = "Exact solution")


plt.plot(x_predict, (y_predict - np.exp(-x_predict)), label = "difference")

epoch = history.epoch
loss = history.history['loss']
plt.plot(epoch, loss)
