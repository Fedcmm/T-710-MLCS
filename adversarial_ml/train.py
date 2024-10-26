import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

import preprocessing

INPUT_SHAPE = (28, 28, 1)
OUTPUT_SHAPE = 10
BATCH_SIZE = 128
EPOCHS = 10
VERBOSE = 1


# From https://www.kaggle.com/code/imdevskp/digits-mnist-classification-using-cnn
def create_model() -> Sequential:
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    x_train, y_train_enc, x_test, y_test_enc = preprocessing.get_dataset()

    model = create_model()
    model.summary()

    model.fit(x_train, y_train_enc,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=VERBOSE,
              validation_split=0.3)

    model.save("mnist_cnn.keras")

    loss, accuracy = model.evaluate(x_test, y_test_enc, verbose=False)
    print(f"Loss: {loss:.6}, Accuracy: {accuracy:.4}")

    y_test = [np.argmax(i) for i in y_test_enc]
    y_pred = [np.argmax(i) for i in model.predict(x_test)]
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    train()