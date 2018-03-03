#!/usr/bin/env python3

#  Python example of Convolutional Neural Network.
#  Please refer primitiv repository for more details.
#
# Usage:
#   $ ./download_data.sh
#   $ python3 ./mnist_cnn.py

import random

import numpy as np

from primitiv import functions as F
from primitiv import initializers as I
from primitiv import optimizers as O
from primitiv import devices as D
from primitiv import Device, Graph, Parameter, Shape

NUM_TRAIN_SAMPLES = 60000
NUM_TEST_SAMPLES = 10000
BATCH_SIZE = 200
NUM_TRAIN_BATCHES = NUM_TRAIN_SAMPLES // BATCH_SIZE
NUM_TEST_BATCHES = NUM_TEST_SAMPLES // BATCH_SIZE
MAX_EPOCH = 100

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

KERNEL_SIZE1 = 5  # should be an odd number
KERNEL_SIZE2 = 5  # ditto
NUM_CHANNELS1 = 8
NUM_CHANNELS2 = 16
PADDING1 = KERNEL_SIZE1 // 2
PADDING2 = KERNEL_SIZE2 // 2

NUM_INPUT_UNITS = (IMAGE_HEIGHT // 4) * (IMAGE_WIDTH // 4) * NUM_CHANNELS2
NUM_HIDDEN_UNITS = 256
NUM_OUTPUT_UNITS = 10


def load_images(filename, n):
    with open(filename, "rb") as ifs:
        ifs.seek(16)  # header
        return (np.fromfile(ifs, dtype=np.uint8, count=n*NUM_INPUT_UNITS) / 255) \
            .astype(np.float32) \
            .reshape((n, IMAGE_HEIGHT, IMAGE_WIDTH))


def load_labels(filename, n):
    with open(filename, "rb") as ifs:
        ifs.seek(8)  # header
        return np.fromfile(ifs, dtype=np.uint8, count=n) \
            .astype(np.uint32)

def main():
    # Loads data
    train_inputs = load_images("data/train-images-idx3-ubyte", NUM_TRAIN_SAMPLES)
    train_labels = load_labels("data/train-labels-idx1-ubyte", NUM_TRAIN_SAMPLES)
    test_inputs = load_images("data/t10k-images-idx3-ubyte", NUM_TEST_SAMPLES)
    test_labels = load_labels("data/t10k-labels-idx1-ubyte", NUM_TEST_SAMPLES)

    dev = D.CUDA(0);
    Device.set_default(dev)
    g = Graph()
    Graph.set_default(g)

    # Parameters of CNNs
    # Shape: {kernel_height, kernel_width, in_channels, out_channels}
    pw_cnn1 = Parameter(
        Shape([KERNEL_SIZE1, KERNEL_SIZE1, 1, NUM_CHANNELS1]),
        I.XavierUniformConv2D())
    pw_cnn2 = Parameter(
        Shape([KERNEL_SIZE2, KERNEL_SIZE2, NUM_CHANNELS1, NUM_CHANNELS2]),
        I.XavierUniformConv2D())

    # Parameters of FC layers
    pw_fc1 = Parameter(Shape([NUM_HIDDEN_UNITS, NUM_INPUT_UNITS]), I.XavierUniform())
    pw_fc2 = Parameter(Shape([NUM_OUTPUT_UNITS, NUM_HIDDEN_UNITS]), I.XavierUniform())
    pb_fc1 = Parameter(Shape([NUM_HIDDEN_UNITS]), I.Constant(0))
    pb_fc2 = Parameter(Shape([NUM_OUTPUT_UNITS]), I.Constant(0))

    # Optimizer
    optimizer = O.SGD(.1)
    optimizer.add(pw_cnn1, pw_cnn2, pw_fc1, pw_fc2, pb_fc1, pb_fc2)

    # Helper lambda to construct the predictor network.
    def make_graph(inputs, train):
        # Input and parameters.
        #x = F.input(Shape([IMAGE_HEIGHT, IMAGE_WIDTH], BATCH_SIZE), inputs)
        x = F.input(inputs)
        w_cnn1 = F.parameter(pw_cnn1)
        w_cnn2 = F.parameter(pw_cnn2)
        w_fc1 = F.parameter(pw_fc1)
        w_fc2 = F.parameter(pw_fc2)
        b_fc1 = F.parameter(pb_fc1)
        b_fc2 = F.parameter(pb_fc2)
        # CNNs
        h_cnn1 = F.relu(F.conv2d(x, w_cnn1, PADDING1, PADDING1, 1, 1, 1, 1))
        h_pool1 = F.max_pool2d(h_cnn1, 2, 2, 0, 0, 2, 2)
        h_cnn2 = F.relu(F.conv2d(h_pool1, w_cnn2, PADDING2, PADDING2, 1, 1, 1, 1))
        h_pool2 = F.max_pool2d(h_cnn2, 2, 2, 0, 0, 2, 2)
        # FC layers
        x_fc = F.dropout(F.flatten(h_pool2), .5, train)
        h_fc = F.dropout(
            F.relu(F.matmul(w_fc1, x_fc) + b_fc1), .5, train)
        return F.matmul(w_fc2, h_fc) + b_fc2

    # Batch randomizer
    ids = list(range(NUM_TRAIN_SAMPLES))

    for epoch in range(MAX_EPOCH):
        # Shuffles sample IDs.
        random.shuffle(ids)

        # Training loop
        for batch in range(NUM_TRAIN_BATCHES):
            print("\rTraining... %d / %d" % (batch + 1, NUM_TRAIN_BATCHES), end="")
            # Makes a minibatch for training.
            inputs = [train_inputs[ids[batch * BATCH_SIZE + i]] for i in range(BATCH_SIZE)]
            labels = [train_labels[ids[batch * BATCH_SIZE + i]] for i in range(BATCH_SIZE)]

            # Constructs the graph.
            g.clear();
            y = make_graph(inputs, True);
            loss = F.softmax_cross_entropy(y, labels, 0)
            avg_loss = F.batch.mean(loss)

            # Dump computation graph at the first time.
            # if epoch == 0 and batch == 0:
            #     print(g.dump("dot"))

            # Implicit forward, backward, and updates parameters.
            optimizer.reset_gradients()
            avg_loss.backward()
            optimizer.update()

        print()

        match = 0

        # Test loop
        for batch in range(NUM_TEST_BATCHES):
            print("\rTesting... %d / %d" % (batch + 1, NUM_TEST_BATCHES), end="")
            # Makes a test minibatch.
            inputs = [test_inputs[batch * BATCH_SIZE + i] for i in range(BATCH_SIZE)]

            # Constructs the graph.
            g.clear()
            y = make_graph(inputs, False)

            # Gets outputs, argmax, and compares them with the label.
            y_val = y.to_list()
            for i in range(BATCH_SIZE):
                maxval = -1e10
                argmax = -1
                for j in range(NUM_OUTPUT_UNITS):
                    v = y_val[j + i * NUM_OUTPUT_UNITS]
                    if v > maxval:
                        maxval = v
                        argmax = j

                if argmax == test_labels[i + batch * BATCH_SIZE]:
                    match += 1

        accuracy = 100.0 * match / NUM_TEST_SAMPLES;
        print("epoch %d: accuracy: %.2f%%" % (epoch, accuracy))

    return 0


if __name__ == "__main__":
    main()
