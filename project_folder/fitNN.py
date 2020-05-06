import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')
n = 100
input_tensor = tf.keras.layers.Input(shape=(1,)) #creates symbolic tensor for input
## Activation function for hidden layers
#activation_function = tf.nn.leaky_relu
#activation_function = tf.nn.tanh
activation_function = tf.nn.sigmoid
#activation_function = tf.nn.elu
#activation_function = tf.nn.swish
#input_layer = tf.keras.layers.Dense(n, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = tf.nn.elu)(input_tensor)
hidden_layer_1 = tf.keras.layers.Dense(10, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = activation_function)(input_tensor)
#hidden_layer_2 = tf.keras.layers.Dense(100, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = activation_function)(hidden_layer_1)
#hidden_layer_3 = tf.keras.layers.Dense(100, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = activation_function)(hidden_layer_2)
output_layer = tf.keras.layers.Dense(1, kernel_initializer= 'GlorotNormal', bias_initializer='zeros',activation = tf.identity)(hidden_layer_1)
model = tf.keras.Model(inputs = input_tensor, outputs = output_layer)

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07)

def mappable_func(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    dy_dx = tape.gradient(y, x)
    return tf.square(dy_dx + y - tf.math.exp(-x) * tf.math.cos(x))



def custom_loss(x):
    def loss(y_true, y_pred):
        differential_loss = tf.math.reduce_sum(tf.map_fn(mappable_func, x_train_tf))
        return differential_loss/n +  tf.square(model(np.asarray([0]))[0][0])
    return loss


x_train_tf = tf.reshape(tf.convert_to_tensor(np.linspace(0,1,n)), (100,1))
model.compile(loss = custom_loss(x_train_tf), optimizer = optimizer,  experimental_run_tf_function = False)
model.summary()

epochs = 10000
history = model.fit(x = x_train_tf, y = x_train_tf, batch_size = n, epochs = epochs)


#plotting 
x_predict = np.linspace(0,4, n*100)
y_predict = model.predict(x_predict)
y_predict = np.reshape(y_predict, (n*100))
#y_true = np.exp(-x_predict)
y_true = np.exp(-x_predict)*np.sin(x_predict)

fig = plt.figure()
plt.plot(x_predict, y_predict, ".")
plt.plot(x_predict, y_true)

epoch = history.epoch
loss = history.history['loss']
fig = plt.figure()
plt.plot(epoch,loss)
