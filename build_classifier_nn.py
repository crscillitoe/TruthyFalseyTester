from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)

    train_filter = np.where((y_train == 0 ) | (y_train == 1))
    test_filter  = np.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_filter], to_categorical(y_train[train_filter])
    x_test , y_test  = x_test[test_filter],   y_test[test_filter]

    print(y_train[0:10])

    X      = x_train.reshape((x_train.shape[0], 28 * 28, ))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28, ))

    model = build_model()
    model.fit(X, y_train, epochs=10, batch_size=512)

    yhat = model.predict(x_test)
    yhat = [round(x[1]) for x in yhat]

    accuracy = accuracy_score(yhat, y_test)
    assert(accuracy > 0.9)
    model.save('1-0-classifier.h5')

def build_model():
    """
    Defines a neural network model for classifying
    images containing either handwritten 1s, or
    handwritten 0s.
    """
    model = Sequential()

    model.add(Dense(40, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('relu'))

    model.add(Dense(40, activation='tanh', kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', use_bias =True))
    model.add(Dense(40, activation='tanh', kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', use_bias =True))

    model.add(Dense(2, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    main()