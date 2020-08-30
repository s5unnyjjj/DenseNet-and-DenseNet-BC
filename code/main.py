from tensorflow import keras
from tensorflow.keras.datasets import mnist

import time
import numpy as np
import matplotlib.pyplot as plt


class DenseNet:
    def __init__(self, name):
        self.growth_rate = 32
        self.block_num = 1
        self.architecture = name
        self.cur_featureMap = 0
        self.theta = 0.5

    def dense_layer(self, model):
        if self.architecture == 'DenseNet':
            x = self.composite_function(model)
            x_cc = keras.layers.concatenate([model, x])

            y = self.composite_function(x_cc)
            y_cc = keras.layers.concatenate([x_cc, y])

            z = self.composite_function(y_cc)
            z_cc = keras.layers.concatenate([y_cc, z])

        if self.architecture == 'DenseNetBC':
            x = self.bottleneck_layer(model)
            x_cc = keras.layers.concatenate([model, x])

            y = self.bottleneck_layer(x_cc)
            y_cc = keras.layers.concatenate([x_cc, y])

            z = self.bottleneck_layer(y_cc)
            z_cc = keras.layers.concatenate([y_cc, z])

        return z_cc

    def composite_function(self, model):
        model = keras.layers.BatchNormalization(momentum=0.5)(model)
        model = keras.layers.Activation('relu')(model)
        model = keras.layers.Conv2D(self.growth_rate, (3, 3), padding='same')(model)
        model = keras.layers.Dropout(rate=0.2)(model)
        return model

    def bottleneck_layer(self, model):
        model = keras.layers.BatchNormalization(momentum=0.5)(model)
        model = keras.layers.Activation('relu')(model)
        model = keras.layers.Conv2D(4*self.growth_rate, (1, 1), padding='same')(model)
        model = keras.layers.Dropout(rate=0.2)(model)

        model = keras.layers.BatchNormalization(momentum=0.5)(model)
        model = keras.layers.Activation('relu')(model)
        model = keras.layers.Conv2D(self.growth_rate, (3, 3), padding='same')(model)
        self.cur_featureMap += 4 * self.growth_rate
        model = keras.layers.Dropout(rate=0.2)(model)

        return model

    def transition_layer(self, model):
        if self.architecture == 'DenseNet':
            model = keras.layers.Conv2D(self.growth_rate, (1, 1), padding='same')(model)
            model = keras.layers.AveragePooling2D((2, 2))(model)

        if self.architecture == 'DenseNetBC':
            model = keras.layers.Conv2D(self.theta * self.cur_featureMap, (1, 1), padding='same')(model)
            model = keras.layers.AveragePooling2D((2, 2))(model)

        return model

    def build_model(self):
        input = keras.layers.Input((28, 28, 1))
        model = keras.layers.Conv2D(256, (7, 7), padding='same', strides=(1, 1))(input)
        self.cur_featureMap += 256

        model = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(model)

        model = self.dense_layer(model)

        for i in range(self.block_num-1):
            model = self.transition_layer(model)
            model = self.dense_layer(model)

        model = keras.layers.BatchNormalization(momentum=0.5)(model)
        model = keras.layers.Activation('relu')(model)
        model = keras.layers.GlobalAveragePooling2D()(model)
        output = keras.layers.Dense(10, activation='softmax')(model)

        res = keras.Model(input, output)
        return res

    def eval(self, model):

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        val_images = train_images[50000:]
        val_labels = train_labels[50000:]
        train_images = train_images[:50000]
        train_labels = train_labels[:50000]

        train_images = train_images.reshape(50000, 28, 28, 1).astype('float32') / 255.0
        val_images = val_images.reshape(10000, 28, 28, 1).astype('float32') / 255.0

        train_rand_idx = np.random.choice(50000, 700)
        val_rand_idx = np.random.choice(10000, 300)

        train_images = train_images[train_rand_idx]
        train_labels = train_labels[train_rand_idx]
        val_images = val_images[val_rand_idx]
        val_labels = val_labels[val_rand_idx]

        train_labels = keras.utils.to_categorical(train_labels)
        val_labels = keras.utils.to_categorical(val_labels)

        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit(train_images, train_labels, batch_size=10, epochs=100, validation_data=(val_images, val_labels))

        fig, loss_ax = plt.subplots()

        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

        acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuracy')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.show()
        return model


if __name__ == '__main__':
    startTime = time.time()
    dense_model = DenseNet('DenseNetBC')
    model = dense_model.build_model()
    model.summary()
    dense_model.eval(model)
    endTime = time.time()
    print("Total Time : ", endTime - startTime)

