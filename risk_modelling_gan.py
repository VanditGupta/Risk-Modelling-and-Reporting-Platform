import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder

# Load policy data
policy_df = pd.read_csv('Datasets/policy_data_geospatial.csv')

# One-hot encode categorical columns
categorical_columns = ['PolicyType']
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(policy_df[categorical_columns])

# Create a new DataFrame with encoded categorical columns
encoded_df = pd.DataFrame(
    encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))
policy_df = policy_df.drop(
    categorical_columns + ['StartDate', 'EndDate'], axis=1)  # Drop date columns
policy_df = pd.concat([policy_df, encoded_df], axis=1)

num_features = policy_df.shape[1]  # Number of columns in policy_df

# Define GAN components


def build_generator():
    model = Sequential([
        Dense(256, activation='relu', input_dim=100),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(num_features, activation='tanh')
    ])
    return model


def build_discriminator():
    model = Sequential([
        Input(shape=(num_features,)),  # Use Input layer to specify the shape
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


# Build and compile GAN
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Train GAN


def train_gan(gan, generator, discriminator, policy_df, epochs=1000, batch_size=128):  # Reduced to 1000 epochs
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_data = generator.predict(noise)
        real_data = policy_df.sample(batch_size).values.astype(np.float32)
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))
        discriminator.train_on_batch(real_data, labels_real)
        discriminator.train_on_batch(generated_data, labels_fake)
        noise = np.random.normal(0, 1, (batch_size, 100))
        labels_gan = np.ones((batch_size, 1))
        gan.train_on_batch(noise, labels_gan)
        print(f"Epoch {epoch + 1} completed.")  # Print all epoch numbers


train_gan(gan, generator, discriminator, policy_df)

# Generate synthetic scenarios
noise = np.random.normal(0, 1, (policy_df.shape[0], 100))
synthetic_scenarios = generator.predict(noise)

# Save generated scenarios to CSV
synthetic_scenarios_df = pd.DataFrame(
    synthetic_scenarios, columns=policy_df.columns)
synthetic_scenarios_df.to_csv('synthetic_scenarios.csv', index=False)

print("Synthetic scenarios generated successfully!")
