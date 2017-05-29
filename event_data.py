import numpy as np
import tensorflow as tf

num_events = 1
seq_len = 2
num_users = 3

data = []

def generate_event_data():
    data = []
    for user in range(num_users):
        user_data = []
        for i in range(seq_len):
            user_data.append([1, i])
        data.append(user_data)

generate_event_data()

batch_count = 0
def next_batch(batch_size):
    def generate_ttes():
        ttes = []
        for user in range(batch_size):
            user_ttes = []
            for i in range(num_events):
                user_ttes.append(1)
            ttes.append(user_ttes)
        return ttes

    def generate_censored():
        u = []
        for user in range(batch_size):
            user_u = []
            for i in range(seq_len):
                if i == seq_len - 1:
                    user_u.append(1.0)
                else:
                    user_u.append(0.0)
            u.append(user_u)
        return u

    print data
    print generate_ttes()
    print generate_censored()
    return tf.constant(data), tf.constant(generate_ttes()), tf.constant(generate_censored())
