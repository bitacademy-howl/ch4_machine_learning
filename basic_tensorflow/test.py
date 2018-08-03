
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import itemfreq

x = tf.random_normal([1])


# Launch the graph in a session
sess = tf.Session()

# Initializes gglobal variables in the graph.
sess.run(tf.global_variables_initializer())

# print(sess.run(x))

x1 = []
y = []
for i in range(100):
    x1.append(i)
    y.append(sess.run(x))

plt.scatter(x1, y)

print(y)
plt.show()

ss = []
ss.append([i//2 for i in y])
print(ss)

itemfreq(ss)
# print(y)