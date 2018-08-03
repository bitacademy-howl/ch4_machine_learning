import tensorflow as tf

# gradient descent :
# 개 요 : 머신러닝에서 알고리즘의 Bias-variance Trade-off 를 최적점에 맞추기 위한
#         모델 최적화의 단계

# 다변량 회귀 모델에서는 어떻게?????
x1_data = [1,2,3,4,5]
x2_data = [1,2,3,4,5]
y_data = [9,14,19,24,29]

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# standard deviation = 1, mean = 0, 표준 정규분포
W1 = tf.Variable(tf.random_normal([1]), name='weight1')
W2 = tf.Variable(tf.random_normal([1]), name='weight2')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = (X1 * W1) + (X2 * W2) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes gglobal variables in the graph.
sess.run(tf.global_variables_initializer())

# 여기는 sampling theory에 맞게 러닝레이트의 2배수 이상??
for step in range(2000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X1:x1_data, X2:x2_data, Y:y_data})
    if step % 10 == 0:
        print(step, cost_val, hy_val)