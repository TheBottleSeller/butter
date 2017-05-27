'''
Churn dat butter
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Global parameters
total_event_types = 5
display_step = 10

# Perf parameters to fuck with
cell_size = 128             # size of weights and biases internal to a lstm cell
n_layers = 3                # layers of dis bitch
training_iters = 100000
batch_size = 128            # number of users to run in a training batch
learning_rate = 0.001
lstm_cell_constructor = rnn.cell.GRUCell
forget_bias = 1.0

# Input and output parameters
'''
Inputs are a tuple of the form [
    type    an enum of the user event
    dt      time since the last event
    value   value of the event at this time
]

Output is of the form [
    [a_1, b_1]    alpha and beta to predict event type 1
    [a_2, b_2]    alpha and beta to predict event type 2
    ...
    [a_n, b_n]    alpha and beta to predict event type (total_event_types)
]
'''
n_input = 3
seq_len = 256   # last x events of a user
X = tf.placeholder(tf.float32, [batch_size, seq_len, n_input])
Y = tf.placeholder(tf.float32, [batch_size, total_event_types, 2])

# Define h initial
H_in = tf.placeholder(tf.float23, [batch_size, cell_size * n_layers])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([cell_size, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def model(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, seq_len, n_input)
    # Required shape: 'seq_len' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'seq_len' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_len, 1)

    # The stacked lstm
    cell = lstm_cell_constructor(cell_size, forget_bias=1.0)
    mcell = rnn.MultiRNNCell([lstm_cell] * n_layers, state_is_tuple=False)

    # Get output
    #Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=H_in)
    # Hr = [batch_size, seq_len, cell_size] = results of each sequence run
    # H = final H weights for each layer
    Hr, H = tf.nn.dynamic_rnn(mcell, X)

    # Transform Hf = [batch_size x seq_len, cell_size]
    # For each recurrent iteration for each user, we have cell_size vector results
    # We need to map the cell_size vector to the expected output of alpha, beta values
    Hf = tf.reshape(Hr, [-1, cell_size])

    # Use a linear NN to map each cell_size input to the alpha beta predictions
    # The output is a single vector that are all the alpha beta laid out in a row
    # I.e. [a_1, b_1, a_2, b_2, ..., a_n, b_n]
    # alpha_betas = [batch_size x seq_len, 2 * total_event_types]
    alpha_betas_raw = layers.linear(Hf, 2 * total_event_types)
    alpha_betas = tf.reshape(alpha_betas_raw, [batch_size x seq_len, total_event_types, 2])
    

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = LSTM_RNN(X, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, seq_len, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, seq_len, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
