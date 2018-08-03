import tensorflow as tf
import matplotlib.pyplot as plt

# gradient descent :
# 개 요 : 머신러닝에서 알고리즘의 Bias-variance Trade-off 를 최적점에 맞추기 위한
#         모델 최적화의 단계

X = [1, 2, 3]
Y = [1, 2, 3]
W = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

W_history = []
cost_history = []

for i in range(-30, 50):
    curr_W = i * 0.1
# for curr_W in range(-3, 5, 0.1):
    curr_cost = sess.run(cost, feed_dict={W:curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)

plt.plot(W_history, cost_history)
plt.show()