#!/usr/bin/env python3

# Sample code to train/test the MNIST dataset:
#   http://yann.lecun.com/exdb/mnist/
#
# The model consists of a full-connected 2-layer (input/hidden/output)
# perceptron with the softmax cross entropy loss.
#
# Usage:
#   $ ./download_data.sh
#   $ python3 ./mnist.py


from primitiv import Device
from primitiv import Graph
from primitiv import Parameter

from primitiv import devices as D
from primitiv import initializers as I
from primitiv import operators as F
from primitiv import optimizers as O

import random
import sys
import numpy as np


NUM_TRAIN_SAMPLES = 60000
NUM_TEST_SAMPLES = 10000
NUM_INPUT_UNITS = 28 * 28
NUM_HIDDEN_UNITS = 800
NUM_OUTPUT_UNITS = 10
BATCH_SIZE = 200
NUM_TRAIN_BATCHES = NUM_TRAIN_SAMPLES // BATCH_SIZE
NUM_TEST_BATCHES = NUM_TEST_SAMPLES // BATCH_SIZE
MAX_EPOCH = 100


def load_images(filename, n):
    with open(filename, "rb") as ifs:
        ifs.seek(16)  # header
        return (np.fromfile(ifs, dtype=np.uint8, count=n*NUM_INPUT_UNITS) / 255) \
            .astype(np.float32) \
            .reshape((n, NUM_INPUT_UNITS))


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

    dev = D.Naive()  # or D.CUDA(gpuid)
    Device.set_default(dev)

    pw1 = Parameter([NUM_HIDDEN_UNITS, NUM_INPUT_UNITS], I.XavierUniform())
    pb1 = Parameter([NUM_HIDDEN_UNITS], I.Constant(0))
    pw2 = Parameter([NUM_OUTPUT_UNITS, NUM_HIDDEN_UNITS], I.XavierUniform())
    pb2 = Parameter([NUM_OUTPUT_UNITS], I.Constant(0))

    optimizer = O.SGD(.5)
    optimizer.add_parameter(pw1)
    optimizer.add_parameter(pb1)
    optimizer.add_parameter(pw2)
    optimizer.add_parameter(pb2)

    def make_graph(inputs, train):
        x = F.input(inputs)

        w1 = F.parameter(pw1)
        b1 = F.parameter(pb1)
        h = F.relu(w1 @ x + b1)

        h = F.dropout(h, .5, train)

        w2 = F.parameter(pw2)
        b2 = F.parameter(pb2)
        return w2 @ h + b2

    ids = list(range(NUM_TRAIN_SAMPLES))

    g = Graph()
    Graph.set_default(g)

    for epoch in range(MAX_EPOCH):
        random.shuffle(ids)

        # Training loop
        for batch in range(NUM_TRAIN_BATCHES):
            print("\rTraining... %d / %d" % (batch + 1, NUM_TRAIN_BATCHES), end="")
            inputs = [train_inputs[ids[batch * BATCH_SIZE + i]] for i in range(BATCH_SIZE)]
            labels = [train_labels[ids[batch * BATCH_SIZE + i]] for i in range(BATCH_SIZE)]

            g.clear()

            y = make_graph(inputs, True)
            loss = F.softmax_cross_entropy(y, labels, 0)
            avg_loss = F.batch.mean(loss)

            optimizer.reset_gradients()
            avg_loss.backward()
            optimizer.update()

        print()

        match = 0

        # Test loop
        for batch in range(NUM_TEST_BATCHES):
            print("\rTesting... %d / %d" % (batch + 1, NUM_TEST_BATCHES), end="")
            inputs = [test_inputs[batch * BATCH_SIZE + i] for i in range(BATCH_SIZE)]

            g.clear()

            y = make_graph(inputs, False)
            y_val = y.to_list()
            for i in range(BATCH_SIZE):
                maxval = -1e10
                argmax = -1
                for j in range(NUM_OUTPUT_UNITS):
                    v = y_val[j + i * NUM_OUTPUT_UNITS]
                    if (v > maxval):
                        maxval = v
                        argmax = j
                if argmax == test_labels[i + batch * BATCH_SIZE]:
                    match += 1

        accuracy = 100.0 * match / NUM_TEST_SAMPLES
        print("\nepoch %d: accuracy: %.2f%%\n" % (epoch, accuracy))


if __name__ == "__main__":
    main()
