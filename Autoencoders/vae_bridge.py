import tensorflow as tf
from tensorflow.keras import Model, layers
from dataclasses import dataclass, field
from typing import Any, List
import numpy as np


@dataclass
class ConfigVAE:
    """
    Data class with the configuration for the Variational Autoencoder architecture.
    All attributes are self-explanatory. However, there are a few important notes:
        - The ``validation_split`` value is ignored if validation data is supplied to the ``fit()`` method.
        - Each value of the ``neurons`` list specifies the number of neurons in each hidden layer. Therefore,
            for multi-layer architectures, this list would contain two or more values.
        - The ``dropout`` rates are optional. However, if they are supplied, an independent rate for each
            hidden layer must be defined.
        - The only mandatory attribute is the number of features of the dataset.
    """
    optimizer: Any = "adam"
    loss: Any = "mean_squared_error"
    metrics: List = field(default_factory=lambda: [])
    callbacks: List = field(default_factory=lambda: [])
    epochs: int = 200
    batch_size: int = 64
    validation_split: float = 0.0
    verbose: int = 1
    activation: str = "relu"
    output_activation: str = "sigmoid"
    neurons: List[int] = field(default_factory=lambda: [10])
    dropout_rates: List[float] = None
    latent_dimension: int = 1
    number_features: int = None


class Sampling(layers.Layer):
    """
    Custom layer which implements the reparameterization trick of the Variational Autoencoder.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        epsilon.set_shape(z_mean.shape)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class KLDivergenceLayer(layers.Layer):
    """
    Custom layer which adds the regularization term of the Kulback-Leibler divergence to the loss value.
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * tf.keras.backend.sum(
            1 + log_var - tf.keras.backend.square(mu) -
            tf.keras.backend.exp(log_var), axis=-1)

        self.add_loss(tf.keras.backend.mean(kl_batch), inputs=inputs)
        return inputs


class VariationalAutoEncoder:
    """
    Implementation of the Variational Autoencoder.

    Attributes:
        _config (ConfigVAE): Data class with the configuration for the Variational Autoencoder architecture.
        _model: Complete Keras model (encoding and decoding), obtained after the fitting process.
        _encoder: Keras model of the encoding side, obtained after the fitting process.
        _decoder: Keras model of the decoding side, obtained after the fitting process.
        _fitted (bool): Boolean flag used to indicate if the ``fit()`` method was already invoked.
    """
    def __init__(self, config: ConfigVAE):
        self._config = config
        self._model = None
        self._encoder = None
        self._decoder = None
        self._fitted = False

    def _create_auto_encoder(self, input_shape):
        """
        Creates the Variational Autoencoder Keras models with the architecture details provided in ``_config``.

        Args:
            input_shape: Input shape of the Variational Autoencoder, based on the number of features of the dataset.

        Returns: Tuple with three Keras models: complete model, encoding and decoding sides.

        """
        x = enc_input = tf.keras.Input(shape=input_shape)

        for i, n in enumerate(self._config.neurons):
            x = layers.Dense(n, activation=self._config.activation)(x)
            if self._config.dropout_rates is not None:
                x = layers.Dropout(rate=self._config.dropout_rates[i])(x)

        z_mean_layer = layers.Dense(self._config.latent_dimension, name="z_mean")
        z_log_var_layer = layers.Dense(self._config.latent_dimension, name="z_log_var")
        z_mean = z_mean_layer(x)
        z_log_var = z_log_var_layer(x)
        z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
        x = [z_mean, z_log_var, Sampling()([z_mean, z_log_var])]

        m_encoder = Model(enc_input, x, name='encoder')
        dec_input = tf.keras.Input(shape=(self._config.latent_dimension,))
        x = dec_input

        for i, n in reversed(list(enumerate(self._config.neurons))):
            x = layers.Dense(n, activation=self._config.activation)(x)
            if self._config.dropout_rates is not None:
                x = layers.Dropout(rate=self._config.dropout_rates[i])(x)

        x = layers.Dense(units=list(filter(None, enc_input.get_shape().as_list()))[0],
                         activation=self._config.output_activation)(x)

        m_decoder = Model(dec_input, x, name='decoder')
        enc_output = m_decoder(m_encoder(enc_input)[2])
        m_global = Model(enc_input, enc_output, name='vae')

        return m_global, m_encoder, m_decoder

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        Fits the Variational Autoencoder model.

        Args:
            x_train: Training data.
            y_train: Target data.
            x_val (optional): Validation training data.
            y_val (optional): Validation target data.
        """
        self._model, self._encoder, self._decoder = self._create_auto_encoder(self._config.number_features)
        self._model.compile(optimizer=self._config.optimizer, loss=self._config.loss, metrics=self._config.metrics)

        fit_args = {
            "epochs": self._config.epochs,
            "batch_size": self._config.batch_size,
            "callbacks": self._config.callbacks,
            "validation_split": self._config.validation_split,
            "verbose": self._config.verbose
        }

        if x_val is not None and y_val is not None:
            fit_args["validation_data"] = (x_val, y_val)

        self._model.fit(x_train, y_train, **fit_args)
        self._fitted = True

    def encode(self, x):
        """
        Encodes new data points with the Variational Autoencoder.

        Args:
            x: Data to be encoded.

        Returns: The encoded representation of ``x``.

        """
        if not self._fitted:
            raise RuntimeError("The fit method must be called before encode.")
        return np.asarray(self._encoder.predict(x))

    def decode(self, x):
        """
        Decodes encoded representations with the Variational Autoencoder.

        Args:
            x: Data to be decoded.

        Returns: The decoded data from the encoded representations supplied in ``x``.

        """
        if not self._fitted:
            raise RuntimeError("The fit method must be called before decode.")
        return self._decoder.predict(x)
