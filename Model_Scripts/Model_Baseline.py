import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.training.rmsprop import RMSPropOptimizer
import os

HEIGHT = 450
WIDTH = 450
DEPTH = 3
NUM_CLASSES = 2
BATCH_SIZE = 32
INPUT_TENSOR_NAME = "inputs_input" # According to Amazon, needs to match the name of the first layer + "_input"
                                   # Workaround for actual known bugs

def keras_model_fn(hyperparameters):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(HEIGHT, WIDTH, DEPTH), activation="relu", name="inputs",
                     padding="same"))
    model.add(MaxPooling2D())
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    opt = RMSPropOptimizer(learning_rate=hyperparameters['learning_rate'], decay=hyperparameters['decay'])

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model


def serving_input_fn(hyperparameters):
    tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_dir=training_dir)


def _input(mode, batch_size, data_dir):
    assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

    if mode == tf.estimator.ModeKeys.TRAIN:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)
    images, labels = generator.next()

    return {INPUT_TENSOR_NAME: images}, labels