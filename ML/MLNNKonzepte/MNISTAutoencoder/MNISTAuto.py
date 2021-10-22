import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Codinggröße
encoding_dim = 32

# Eingangsbilder 28x28=784
input_img = keras.Input(shape=(784,))

#Encoder-Schichten
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_img)
#Decoder-Schichten
decoded = keras.layers.Dense(784, activation='sigmoid')(encoded)

#Model für Autoencoder
autoencoder = keras.Model(input_img, decoded)
#Model für Encoder
encoder = keras.Model(input_img, encoded)

#Model für Decoder
encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

#Damit werden Encoder/Decoder mit compiliert
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#Daten anpassen
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

#Training
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#Ein paar Bilder encodieren und decodieren
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#Anzeigen von ein paar Eingangsdaten mit Decodings
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #Decoding
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()