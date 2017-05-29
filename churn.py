'''
Churn dat butter
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from event_data import num_events, next_batch, num_users, seq_len
from loss_functions import loglik_discrete, betapenalty

# Global parameters
display_step = 10

# Perf parameters to fuck with
cell_size = 128             # size of weights and biases internal to a lstm cell
n_layers = 3                # layers of dis bitch
batch_size = num_users      # number of users to run in a training batch
learning_rate = 0.001
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
    [a_n, b_n]    alpha and beta to predict event type (num_events)
]
'''
X = tf.placeholder(tf.float32, [batch_size, seq_len, num_events])
TTEs = tf.placeholder(tf.float32, [batch_size, seq_len, num_events])   # tte for all events for each timestep
Y_ = tf.reshape(TTEs, [batch_size * seq_len, num_events])
#U_ = tf.zer(tf.float32, [batch_size, seq_len])
U_ = tf.zeros([batch_size * seq_len, seq_len])

def loss_func(alpha_betas):
    loss = 0
    for i in range(num_events):
        a = tf.nn.softplus(tf.slice(alpha_betas, [0, 2*i], [-1, 1]))
        b = tf.nn.elu(tf.slice(alpha_betas, [0, 2*i+1], [-1, 1]))
        u_ = tf.slice(U_, [0, i], [-1, 1])
        y_ = tf.slice(Y_, [0, i], [-1, 1])
        loss += -tf.reduce_mean(loglik_discrete(a, b, y_, u_)) + betapenalty(b)
    return loss


# Define h initial
#H_in = tf.placeholder(tf.float32, [batch_size, cell_size * n_layers])

# The stacked lstm
mcell = rnn.MultiRNNCell([rnn.GRUCell(cell_size) for _ in range(n_layers)], state_is_tuple=False)

# Get output
#Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=H_in)
# Hr = [batch_size, seq_len, cell_size] = results of each sequence run
# H = final H weights for each layer [n_layers, cell_size]
Hr, _ = tf.nn.dynamic_rnn(mcell, X, dtype=tf.float32)

# Transform Hf = [batch_size x seq_len, cell_size]
# For each recurrent iteration for each user, we have cell_size vector results
# We need to map the cell_size vector to the expected output of alpha, beta values
Hf = tf.reshape(Hr, [-1, cell_size])

# Use a linear NN to map each cell_size input to the alpha beta predictions
# The output is a single vector that are all the alpha beta laid out in a row
# I.e. [a_1, b_1, a_2, b_2, ..., a_n, b_n]
# alpha_betas = [batch_size x seq_len, 2 * num_events]
alpha_betas = tf.contrib.layers.linear(Hf, 2 * num_events)

loss = loss_func(alpha_betas)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

train_step = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    batch_x, batch_ttes, batch_u = next_batch(batch_size)
    # sess.run([train_step], feed_dict={
    #     X: batch_x,
    #     TTEs: batch_ttes,
    #     U_: batch_u
    # })
    print("Optimization Finished!")
