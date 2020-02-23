import tensorflow as tf
import numpy as np

def create_noise(num_movies, num_months):
    noise = np.empty([num_months, num_movies, num_movies])
    identity = np.identity(num_movies)
    diagonal_noise_mask = np.ones([num_movies,num_movies])-identity
    for m in range(num_months):
        noise_month = np.random.random_integers(970,1030,num_movies)/1000. * diagonal_noise_mask + identity
        noise[m, :, :] = noise_month
    return noise

def create_sample_data(num_movies, num_months):
    signal = np.array([
        [1, 0.93, 0.25, 0.04, 0.37, 0.72],
        [np.nan, 1, 0.33, 0.28, 0.27, 0.7],
        [0.27, 0.33, 1, 0.8, 0.68, 0.41],
        [0.08, 0.08, 0.84, 1, 0.55, 0.15],
        [0.45, np.nan, 0.62, 0.47, 1, 0.5],
        [0.74, 0.66, 0.4, 0.19, 0.49, 1]
    ])
    noise = create_noise(num_movies, num_months)
    data = np.repeat(signal[np.newaxis,:, :], num_months, axis=0) * noise
    return data

'''
    signal = np.array([
        [1, 0.93, 0.25, 0.04, 0.37, 0.72],
        [0.86, 1, 0.33, 0.28, 0.27, 0.7],
        [0.27, 0.33, 1, 0.8, 0.68, 0.41],
        [0.08, 0.08, 0.84, 1, 0.55, 0.15],
        [0.45, 0.34, 0.62, 0.47, 1, 0.5],
        [0.74, 0.66, 0.4, 0.19, 0.49, 1]
    ])
'''


num_movies = 6
num_months = 150
data = create_sample_data(num_movies, num_months)


class Encoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim):
    super(Encoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu,
      kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.sigmoid
    )

  def call(self, input_features):
    activation = self.hidden_layer(input_features)
    return self.output_layer(activation)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim):
    super(Decoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu,
      kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=original_dim,
      activation=tf.nn.sigmoid
    )

  def call(self, code):
    activation = self.hidden_layer(code)
    return self.output_layer(activation)

class Autoencoder(tf.keras.Model):
  def __init__(self, intermediate_dim, original_dim):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder(intermediate_dim=intermediate_dim)
    self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

  def call(self, input_features):
    code = self.encoder(input_features)
    reconstructed = self.decoder(code)
    return reconstructed

def loss(model, original):
  reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
  return reconstruction_error

def train(loss, model, opt, original):
  with tf.GradientTape() as tape:
    gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


learning_rate = 0.01
epochs = 500
batch_size = 10
iterations = int(epochs*num_months/batch_size)

autoencoder = Autoencoder(intermediate_dim=num_movies, original_dim=num_movies*num_movies)

opt = tf.optimizers.Adam(learning_rate=learning_rate)

training_features = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
training_features = training_features.astype('float32')

#training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
#training_dataset = training_dataset.batch(batch_size)
#training_dataset = training_dataset.shuffle(training_features.shape[0])
#training_dataset = training_dataset.prefetch(batch_size * 4)

def replace_na(ndarry):
    random_replace = np.vectorize(lambda x: np.random.randint(2) if np.isnan(x) else x)
    return random_replace(ndarry)

def fetch_batch(training_features, batch_size):
    raw_batch_features = training_features[np.random.choice(training_features.shape[0], batch_size, replace=False), :]
    batch_features_na_filled = replace_na(raw_batch_features)
    return tf.convert_to_tensor(batch_features_na_filled)

writer = tf.summary.create_file_writer('tmp')

with writer.as_default():
  with tf.summary.record_if(True):
    for epoch in range(iterations):
        batch_features = fetch_batch(training_features, batch_size)
        train(loss, autoencoder, opt, batch_features)
        loss_values = loss(autoencoder, batch_features)
        original = tf.reshape(batch_features, (batch_features.shape[0], num_movies, num_movies, 1))
        reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], num_movies, num_movies, 1))


#        tf.summary.scalar('loss', loss_values, step=step)
#        tf.summary.image('original', original, max_outputs=10, step=step)
#        tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
#        tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)

'''
      for step, batch_features in enumerate(training_dataset):
        train(loss, autoencoder, opt, batch_features)
        loss_values = loss(autoencoder, batch_features)
        original = tf.reshape(batch_features, (batch_features.shape[0], num_movies, num_movies, 1))
        reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], num_movies, num_movies, 1))
        tf.summary.scalar('loss', loss_values, step=step)
        tf.summary.image('original', original, max_outputs=10, step=step)
        tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
        original = tf.reshape(batch_features, (batch_features.shape[0], num_movies, num_movies, 1))
'''

def make_pred(training_features, num_movies, num_months):
    return autoencoder(tf.constant(replace_na(training_features))).numpy().reshape(num_months, num_movies, num_movies)












#from sklearn.experimental import enable_iterative_imputer
#from sklearn.imputer import IterativeImputer



# Deciding how many nodes wach layer should have
n_nodes_inpl = 784  #encoder
n_nodes_hl1  = 32  #encoder
n_nodes_hl2  = 32  #decoder
n_nodes_outl = 784  #decoder


size_input = 3
size_hidden = 20
size_output = 1

X_train = np.random.randn(800, size_input)
X_test = np.random.randn(200, size_input)
y_train = np.random.randn(800)
y_test = np.random.randn(200)
