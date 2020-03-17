import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#tf.compat.v1.disable_eager_execution()

# Model Parameters
#input_tensor = tf.keras.layers.Input(shape=[1])
input_shape = [1]
input_n = 100
hidden_1 = 10

# Define model architecture
def build_model(layers):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layers[0], input_shape = input_shape, activation = tf.nn.sigmoid))
    for l in range(1,len(layers)):
        model.add(tf.keras.layers.Dense(layers[l] , activation = tf.nn.sigmoid))

    # Initialize weights and biases using Xavier initialization, look at README
    tf.keras.initializers.GlorotNormal
    #I = tf.keras.layers.Input((100,1))
    return model

# Define custom loss
def custom_loss_function(Inputs_x):
    def loss(y_true, y_pred):
        print("AHHH")
        print(y_pred)
        print(Inputs_x)
        dy_dx = tf.keras.backend.gradients(y_pred,Inputs_x)
        print(dy_dx)
        return tf.keras.backend.mean((dy_dx + Inputs_x)**2)
    return loss

def custom_function(input_tensor):
    def loss(y_true, y_pred):
        #with tf.GradientTape() as tape:
        #    tape.watch(input_tensor)
        #    y = model(input_tensor)
        #dy_dx = tape.gradient(y, model.inputs)
        #dy_dx = tf.gradients(y_pred, model.inputs)
        #return tf.keras.backend.mean((dy_dx + model.inputs)**2)
        return (y_pred**2)
    return loss

# Build model
layers = [100, 10, 1]
model = build_model(layers)

# Set optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss = custom_function(model), optimizer=optimizer,  experimental_run_tf_function=False)




# Visualize NN
model.summary()
x_train = np.linspace(0, 5, 100)
x_train = np.reshape(x_train, newshape = (100, 1))
#Training the neural network
epochs = 1000
history = model.fit(x = x_train, y= x_train, epochs = epochs)


#Testing prediction
x_predict = np.linspace(0, 10, 1000)
y_predict = model.predict(x_predict)
fig = plt.figure()
plt.plot(x_predict, y_predict, ".", label = "Prediction $y$")
plt.plot(x_predict, np.exp(-x_predict))

