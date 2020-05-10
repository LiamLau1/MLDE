import tensorflow as tf

class Base_NNModel():
    """
    Provides an interface to the NN model.
    """
    def __init__(self):
        pass


    def get_parameters(self, filename=None):
        """
        Get the model parameters.
        If filename is provided, it will return the model parameters of the checkpoint. The model is not changed.
        If no arguments are given, it will return the current model parameters.
        Returns a dictionary, comprising of parameter identifiers as keys and numpy arrays as data containers.
        Weights and biases are supposed to have "weight" or "bias" appear in their ID string!

        Args:
            filename: string of the checkpoint, of which the parameters should be returned.
        Return:
            python dictionary of string parameter IDs mapped to numpy arrays.
        """
        raise NotImplementedError("Override this function!")


    def set_parameters(self, parameter_dict):
        """
        Set the model parameters.
        The input dictionary must fit the model parameters!

        Args:
            parameter_dict: python dictionary, mapping parameter id strings to numpy array parameter values.
        """
        raise NotImplementedError("Override this function!")


    def calc_loss(self):
        """
        Calculates the loss of the NN.

        Return:
            The loss, based on the parameters loaded into the model.
        """
        raise NotImplementedError("Override this function!")


class Tensorflow_NNModel(Base_NNModel):
    def __init__(self, model, trigger_fn, filename, number_of_steps = 2):
        print("Build Tensorflow Model ...")
        super(Tensorflow_NNModel, self).__init__()
        print("Setting up Model ...")
        self.number_of_steps = number_of_steps # maximum number of iteration steps per evaluation
        self.model = model
        # doesn't matter what trigger_fn input is as they are not used
        self.total_loss = trigger_fn(1,1)
        print("Initializing Parameters...")
        self.parameter = self._tf_params_to_numpy()
        #TODO: make dict of numpy arrays from self.parameter
        print("Done.")


    def get_parameters(self, filename=None):
        if filename is None:
            return self.parameter
        else:
            self.model.load_weights(filename)
            tmp_params = self._tf_params_to_numpy()
            # restore old state
            self.set_parameters(self.parameter)
            self.parameter = tmp_params
            return self.parameter


    def set_parameters(self, parameter_dict):
        for var in self.model.trainable_variables:
            var.assign(parameter_dict[var.name[:-2]])

    def calc_loss(self, trigger_fn, x):
        average_loss=0
        #print(trigger_fn(tf.convert_to_tensor(x)))
        """
        for i in range(self.number_of_steps):
            current_loss = trigger_fn(tf.convert_to_tensor(x)) # unsure # needs to recall it
            # problem is trigger_fn stays the same
            #print(np.argmax(label,axis=1))
            average_loss += current_loss
        average_loss /= self.number_of_steps
        """
        average_loss = trigger_fn(1,1) #trigger_fn(tf.convert_to_tensor(x))
        print("Average Loss: "+str(average_loss))
        return average_loss


    def _tf_params_to_numpy(self):
        new_param = dict()
        for var in self.model.trainable_variables:
            new_param[var.name[:-2]] = var.numpy()
        return new_param

