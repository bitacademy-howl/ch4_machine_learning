import tensorflow as tf
import matplotlib.pyplot as plt

# gradient descent :
# 개 요 : 머신러닝에서 알고리즘의 Bias-variance Trade-off 를 최적점에 맞추기 위한
#         모델 최적화의 단계

X = [1, 2, 3]
Y = [1, 2, 3]
W = tf.Variable(-3.0) #5 일 경우
W = tf.Variable(5.0) #5 일 경우

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)