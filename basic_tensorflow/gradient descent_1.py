import tensorflow as tf

# # gradient descent :
# 개 요 : 머신러닝에서 알고리즘의 Bias-variance Trade-off 를 최적점에 맞추기 위한
#         모델 최적화의 단계

x_train = [1, 2, 3]
y_train = [1, 2, 3]

# standard deviation = 1, mean = 0, 표준 정규분포
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 실제 모델의 트레이닝 데이터가 정규분포의 확률을 따르고 이에 대한 bias 도 표준정규분포의 확률을 따른다고 가정
hypothesis = x_train * W + b

# Our hypothesis XW+b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# 위의 데이터는 선형회귀 모델을 따르므로....(1변량 모델)
# y = ax + b 라는 함수로 모델링 할 수 있고, 그 a와 b를 가우시안 분포를 따르는 변수들과 대입하여 보면서
# 최적의 모델을 찾을 수 있다.

# 다변량 모델에서는 cost = SSE (각각의 편차의 합 / 변량의 갯수) 로 정의 될 수 있고, 이를 최소화 하는
# 모델을 찾기 위해 아래의 두 줄의 코드를 사용할 수 있다.

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes gglobal variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))


# # ex.2 #################################################################################################################
#
# import tensorflow as tf
#
# # gradient decision :
# # 개 요 : 머신러닝에서 알고리즘의 Bias-variance Trade-off 를 최적점에 맞추기 위한
# #         모델 최적화의 단계
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# # standard deviation = 1, mean = 0, 표준 정규분포
# W = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# hypothesis = X * W + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# # Minimize
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
#
# # Launch the graph in a session
# sess = tf.Session()
#
# # Initializes gglobal variables in the graph.
# sess.run(tf.global_variables_initializer())
#
# for step in range(2001):
#     cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1,2,3,4,5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]})
#     if step % 20 == 0:
#         print(step, cost_val, W_val, b_val)