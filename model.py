"""
Module for all network models with tensorflow dependency
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers, initializers
from typing import Optional, Union, Tuple
from sklearn.linear_model import LinearRegression


class NotInitialized(Exception):
    def __init__(self, message):
        super().__init__(message)


class ActivityPredictor(keras.Model):
    """
    Simple network for non-linear prediction of calcium activity
    """

    def __init__(self, n_units: int, n_conv: int, drop_rate: float, input_length: int, activation: str):
        """
        Creates a new NetNavigator
        :param n_units: The number of units in each dense layer
        :param n_conv: The number of units in each initial convolutional layer
        :param drop_rate: The drop-out rate during training
        :param input_length: The length (across time) of inputs to the network (sets conv filter size)
        :param activation: The activation function to use
        """
        super(ActivityPredictor, self).__init__()
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("drop_rate has to be between 0 and 1")
        if n_units < 1:
            raise ValueError("Need at least one unit in each dense layer")
        if n_conv < 1:
            raise ValueError("Need at least one convolutional unit")
        self._n_units: int = n_units
        self._n_conv: int = n_conv
        self.input_length: int = input_length
        self._activation: str = activation
        self._drop_rate: float = drop_rate
        self.l1_sparsity: float = 2e-4  # sparsity constraint on weight vectors
        self.learning_rate: float = 1e-3
        # optimizer and loss functions
        self.optimizer: Optional[keras.optimizers.Optimizer] = None
        self.loss_fn: Optional[keras.losses.Loss] = None
        # training metrics
        self.rmse: Optional[keras.metrics.Metric] = None
        self._initialized: bool = False
        # output placeholders
        self._out: Optional[keras.layers.Layer] = None
        # layers
        self._conv_layer: Optional[keras.layers.Layer] = None  # Convolutional layer
        self._drop_cl: Optional[keras.layers.Layer] = None  # Dropout of convolutional layer
        self._deep_1: Optional[keras.layers.Layer] = None  # First deep layer
        self._drop_d1: Optional[keras.layers.Layer] = None  # Dropout of first deep layer
        self._deep_2: Optional[keras.layers.Layer] = None  # Second deep layer
        self._drop_d2: Optional[keras.layers.Layer] = None  # Dropout of second deep layer
        self._flatten: Optional[keras.layers.Layer] = None
        # model-specific cash field used during derivative calculation
        self.part_tensor_1: Optional[np.ndarray] = None

    def setup(self) -> None:
        """
        Initializes the model, resetting weights
        """
        # processing
        self._conv_layer = layers.Conv1D(filters=self.n_conv,
                                         kernel_size=self.input_length,
                                         # set kernel size = input size => computes dot product
                                         use_bias=False,  # for simplicity omit bias from convolutional layers
                                         padding='valid',
                                         activation=None,  # make convolutional layers linear
                                         kernel_initializer=initializers.GlorotUniform(),
                                         kernel_regularizer=regularizers.l1(self.l1_sparsity),
                                         strides=1, name="Convolution")
        self._flatten = layers.Flatten()
        self._drop_cl = layers.Dropout(self.drop_rate)
        self._deep_1 = layers.Dense(units=self.n_units, activation=self.activation,
                                    kernel_initializer=initializers.GlorotUniform(),
                                    kernel_regularizer=regularizers.l1(self.l1_sparsity), name="Deep1")
        self._drop_d1 = layers.Dropout(self.drop_rate)
        self._deep_2 = layers.Dense(units=self.n_units, activation=self.activation,
                                    kernel_initializer=initializers.GlorotUniform(),
                                    kernel_regularizer=regularizers.l1(self.l1_sparsity), name="Deep2")
        self._drop_d2 = layers.Dropout(self.drop_rate)
        # output: This is just one value, that should predict the calcium response at the current time
        self._out = layers.Dense(1, activation=None, name="Out")
        # create our optimizer and loss functions
        self.optimizer = keras.optimizers.Adam(self.learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()
        # create training metric
        self.rmse = keras.metrics.RootMeanSquaredError()
        self._initialized = True

    def get_output(self, inputs: np.ndarray) -> float:
        """
        Returns the output value given the model inputs
        :param inputs: batchsize x input_length x n_regressors (the channels)
        :return: 1 output value corresponding to the calcium response
        """
        self.check_input(inputs)
        out = self(inputs)
        return out.numpy().ravel()

    def clear_model(self) -> None:
        """
        Clears and uninitializes the model
        """
        self._conv_layer = None  # Convolutional layer
        self._drop_cl = None  # Dropout of convolutional layer
        self._deep_1 = None  # First deep layer
        self._drop_d1 = None  # Dropout of first deep layer
        self._deep_2 = None  # Second deep layer
        self._drop_d2 = None  # Dropout of second deep layer
        self._out = None
        self._initialized = False

    def check_init(self) -> None:
        if not self._initialized:
            raise NotInitialized("Model not initialized. Call setup or load.")

    def check_input(self, inputs) -> None:
        if inputs.shape[1] != self.input_length:
            raise ValueError("Input length across time different than expected")

    @tf.function
    def call(self, inputs: Union[np.ndarray, tf.Tensor], training: Optional[bool] = None, mask=None) -> tf.Tensor:
        if training is None:
            training = False
        self.check_init()
        inputs = self._conv_layer(inputs, training=training)
        inputs = self._flatten(inputs)
        inputs = self._drop_cl(inputs, training=training)
        inputs = self._deep_1(inputs, training=training)
        inputs = self._drop_d1(inputs, training=training)
        inputs = self._deep_2(inputs, training=training)
        inputs = self._drop_d2(inputs, training=training)
        return self._out(inputs)

    @tf.function
    def train_step(self, btch_inputs: Union[np.ndarray, tf.Tensor], btch_labels: Union[np.ndarray, tf.Tensor]):
        """
        Runs one training step on the model
        :param btch_inputs: The inputs of the batch
        :param btch_labels: True values of the calcium response
        """
        with tf.GradientTape() as tape:
            pred = self(btch_inputs, training=True)
            loss = self.loss_fn(btch_labels, pred)
            # add any extra losses created during foward pass
            # NOTE: I'm not entirely clear if this is really required when the only extra losses
            # are kernel regularizations, however the values match the absolute weights exactly
            # i.e. they are the L1 loss
            loss += sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.rmse(btch_labels, pred)

    def reset_metrics(self) -> None:
        """
        Resets training accuracy metrics of the object
        """
        self.check_init()
        self.rmse.reset_states()

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def drop_rate(self) -> float:
        return self._drop_rate

    @property
    def n_units(self) -> int:
        return self._n_units

    @property
    def n_conv(self) -> int:
        return self._n_conv

    @property
    def conv_layer(self) -> Optional[keras.layers.Layer]:
        return self._conv_layer

    @property
    def deep_1(self) -> Optional[keras.layers.Layer]:
        return self._deep_1

    @property
    def deep_2(self) -> Optional[keras.layers.Layer]:
        return self._deep_2

    @property
    def flatten(self) -> Optional[keras.layers.Layer]:
        return self._flatten

    @property
    def out(self) -> Optional[keras.layers.Layer]:
        return self._out


def train_model(mdl: ActivityPredictor, tset: tf.data.Dataset, n_epochs: int, datacount: int) -> None:
    """
    Trains the model over n epochs. NOTE: The dataset passed should be generated from a shuffle with
    reshuffle_each_iteration set to true to ensure that data is presented in shuffled order in each
    training epoch
    :param mdl: The model to train
    :param tset: The dataset with training data
    :param n_epochs: The number of epochs to train for
    :param datacount: Unused
    """
    for e in range(n_epochs):
        for inp, outp in tset:
            mdl.train_step(inp, outp)


def get_standard_model(hist_steps: int) -> ActivityPredictor:
    """
    Creates and returns an activity predictor instance with standard parameters
    found through a hyper-parameter search
    :param hist_steps: The number of history steps in the model
    """
    m = ActivityPredictor(64, 80, 0.5, hist_steps, "swish")
    m.learning_rate = 1e-3
    m.l1_sparsity = 1e-3
    m.setup()
    return m
