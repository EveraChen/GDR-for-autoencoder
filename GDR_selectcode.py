# importing libs
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from scipy.special import comb, perm

# for reproducing reslut
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(3)

# Autoencoder(2，2)
M = 8
m = 4
k = np.log2(comb(M, m))
k = int(k)
Message = 2**k
n_channel = 7
R = k / n_channel
print('M:', M, 'k:', k, 'n:', n_channel)

# code8_4 = np.array([[1/2,1/2,0,0,0,0,0,0],[1/2,0,1/2,0,0,0,0,0],[1/2,0,0,1/2,0,0,0,0],[1/2,0,0,0,1/2,0,0,0],[1/2,0,0,0,0,1/2,0,0],[1/2,0,0,0,0,0,1/2,0],[1/2,0,0,0,0,0,0,1/2],[0,1/2,1/2,0,0,0,0,0],[0,1/2,0,1/2,0,0,0,0],[0,1/2,0,0,1/2,0,0,0],[0,1/2,0,0,0,1/2,0,0],[0,1/2,0,0,0,0,1/2,0],[0,1/2,0,0,0,0,0,1/2],[0,0,1/2,1/2,0,0,0,0],[0,0,1/2,0,1/2,0,0,0],[0,0,1/2,0,0,1/2,0,0]])
#code8_4=np.array([[1/3,1/3,1/3,0,0,0,0,0],[1/3,1/3,0,1/3,0,0,0,0],[1/3,1/3,0,0,1/3,0,0,0],[1/3,1/3,0,0,0,1/3,0,0],[1/3,1/3,0,0,0,0,1/3,0],[1/3,1/3,0,0,0,0,0,1/3],[0,1/3,1/3,1/3,0,0,0,0],[0,1/3,1/3,0,1/3,0,0,0],[0,1/3,1/3,0,0,1/3,0,0],[0,1/3,1/3,0,0,0,1/3,0],[0,1/3,1/3,0,0,0,0,1/3],[0,0,1/3,1/3,1/3,0,0,0],[0,0,1/3,1/3,0,1/3,0,0],[0,0,1/3,1/3,0,0,1/3,0],[0,0,1/3,1/3,0,0,0,1/3],[0,0,0,1/3,1/3,1/3,0,0],[0,0,0,1/3,1/3,0,1/3,0],[0,0,0,1/3,1/3,0,0,1/3],[0,0,0,0,1/3,1/3,1/3,0],[0,0,0,0,1/3,1/3,0,1/3],[1/3,0,1/3,1/3,0,0,0,0],[1/3,0,0,1/3,1/3,0,0,0],[1/3,0,0,0,1/3,1/3,0,0],[1/3,0,0,0,0,1/3,1/3,0],[1/3,0,0,0,0,0,1/3,1/3],[0,1/3,0,1/3,1/3,0,0,0],[0,1/3,0,0,1/3,1/3,0,0],[0,1/3,0,0,0,1/3,1/3,0],[0,1/3,0,0,0,0,1/3,1/3],[0,0,1/3,0,1/3,1/3,0,0],[0,0,1/3,0,0,1/3,1/3,0],[0,0,1/3,0,0,0,1/3,1/3]])

import scipy.io as sio

load_fn = 'code64.mat'
load_data = sio.loadmat(load_fn)
code8_4 = load_data['y2']

print(code8_4)



N = 20000

# generating data of size N
label = np.zeros(N)
for i in range(0, N):
    label[i] = np.random.choice(Message, replace=False, p=None)
# generate a N*m random number matrix in [0，M)

# creating bits encoded vectors
data = []
for i in range(0, N):
    temp = np.zeros(M)
    temp = code8_4[int(label[i]), :]
    data.append(temp)

data = np.array(data)
print(data.shape)

# defining autoencoder and it's layer
# Transmitter
input_signal = Input(shape=(M,))
#encoded01 = Dense(M, activation='linear')(input_signal)
encoded = Dense(M, activation='relu')(input_signal)  # Transmitter: signal input
encoded1 = Dense(n_channel, activation='linear')(encoded)  # Transmitter: n_Channel transform
#encoded2 = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded1)    # from Berke
encoded2 = Lambda(lambda x: np.sqrt(n_channel)*K.l2_normalize(x, axis=1))(encoded1)  #from T O'Shea
# Transmitter: normalization

# Channel: AWGN
SNR_train = 10  # coverted SNR in dB
EbNo_train = 1 / 2 / k * 10.0 ** (SNR_train / 10.0)
# EbNo_train = 5.01187  # coverted 7 db of EbNo
encoded3 = GaussianNoise(np.sqrt(1 / (2 * R * EbNo_train)))(encoded2)
# Training the autoencoder at 7dB, while it is used in SNR range [-4 8.5] below.

# Receiver
#decoded01= Dense(M, activation='linear')(encoded3)
decoded = Dense(M, activation='relu')(encoded3)
decoded1 = Dense(M, activation='softmax')(decoded)
# output is the probability of all elements suming as 1

# From TX to RX
autoencoder = Model(input_signal, decoded1)
# adam = Adam(lr=0.001)  # learning rate
autoencoder.compile(optimizer='adam',
#                    loss='mean_squared_error')
                   loss='categorical_crossentropy')
# Optimizer, Loss function mean_squared_error (categorical_crossentropy)
# printing summary of layers and it's trainable parameters
print(autoencoder.summary())

# traning auto encoder
autoencoder.fit(data, data,
                epochs=150,
                batch_size=45)

# saving keras model
from keras.models import load_model

# if you want to save model then remove below comment
# autoencoder.save('autoencoder_v_best.model')

# making encoder from full autoencoder
encoder = Model(input_signal, encoded2)

# making decoder from full autoencoder
encoded_input = Input(shape=(n_channel,))

deco = autoencoder.layers[-2](encoded_input)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)

# generating data for checking BER
# if you're not using t-sne for visulation than set N to 70,000 for better result
# for t-sne use less N like N = 1500
N = 1000000

test_label = np.zeros(N)
for i in range(0, N):
    test_label[i] = np.random.choice(Message, replace=False, p=None)

test_data = []
for i in range(0, N):
    temp = np.zeros(M)
    temp = code8_4[int(test_label[i]), :]
    test_data.append(temp)
test_data = np.array(test_data)

# calculating BER
# this is optimized BER function so it can handle large number of N
# previous code has another for loop which was making it slow
def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


SNR = list(frange(0, 26, 5))  # SNR range in dB [-15,10]/5
ber = [None] * len(SNR)  # define BER
for n in range(0, len(SNR)):
    EbNo = 1 / 2 / k * 10.0 ** (SNR[n] / 10.0)
    noise_std = np.sqrt(1 / (2 * R * EbNo))
    noise_mean = 0
    no_errors = 0
    nn = N
    noise = noise_std * np.random.randn(nn, n_channel)

    encoded_signal = encoder.predict(test_data)
    final_signal = encoded_signal + noise
    pred_final_signal = decoder.predict(final_signal)

    # BER function
    # pred_output = np.argmax(pred_final_signal, axis=1)
    pred_output = np.argsort(pred_final_signal, axis=1)
    pred_outputmax = pred_output[:, (M - m):M]
    pred_label_sort = np.sort(pred_outputmax, axis=1)

    test_datasort = np.argsort(test_data, axis =1)
    test_datamax = test_datasort[:, (M - m):M]
    test_label_sort = np.sort(test_datamax, axis=1)

    no_errors = (pred_label_sort != test_label_sort)
    no_errors = no_errors.astype(int)

    error = 0
    for ii in range(0, N):
        error += min(1, no_errors[ii, :].sum())
    ber[n] = error / nn
    print('SNR:', SNR[n], 'BER:', ber[n])
    # use below line for generating matlab like matrix which can be copy and paste for plotting ber graph in matlab
    # print(ber[n], " ",end='')

# ploting ber curve
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

from scipy import interpolate

plt.plot(SNR, ber, 'bo', label='Autoencoder(2,2)')
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right', ncol=1)

print('SNR：',  SNR)
print('BER:', ber)
# for saving figure remove below comment
# plt.savefig('AutoEncoder_2_2_constrained_BER_matplotlib')
plt.show()
