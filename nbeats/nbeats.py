import tensorflow as tf
from revin import RevIN
# Create NBeatsBlock
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self,
                input_size: int,
                theta_size: int,
                horizon: int,
                n_neurons: int,
                n_layers: int,
                useRevIN=False,#extra parameter useRevIN is added which is False by default
                **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        if useRevIN==True: 
            revinlayer = RevIN() #declaring revin layer

        # Block contains stack of 4 fully connected layers each has ReLU activation
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        # Output of block is a theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs): # the call method is what runs when the layer is called 
        x = inputs 
        for layer in self.hidden: # pass inputs through each hidden layer 
            x = layer(x)
        theta = self.theta_layer(x) 
        # Output the backcast and forecast from theta
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast