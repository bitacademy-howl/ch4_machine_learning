import tensorflow as tf
# tf.constant 객체 생성
hello = tf.constant("hello, Tensorflow")


# 세션객체 생성
sess = tf.Session()


# 세션객체의 RUN 메서드
print(sess.run(hello))


a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

print(type(node1))
node3 = tf.add(node1, node2)

print("그냥 프린트 : ", node1, node2, node3)

sess = tf.Session()
print("sess.run(node1, node2) : ", sess.run([node1, node2]))

print(sess.run(node3))

# 그냥 리스트
print(type(sess.run([node1, node2])))


print("3. ##########################################################################")
# 다음은 텐서플로우 Placeholder 에 대해서 알아보겠다.
# 자료형이라고 하는게 맞는지는 모르겠지만.. 어쨌든.. placeholder 자료형은 조금 특이하다.
# 선언과 동시에 초기화 하는 것이 아니라 일단 선언 후 그 다음 값을 전달한다. 따라서 반드시 실행 시 데이터가 제공되어야 한다.
# 여기서 값을 전달한다고 되어 있는데 이는 데이터를 상수값을 전달함과 같이 할당하는 것이 아니라 다른 텐서(Tensor)를
# placeholder에 맵핑 시키는 것이라고 보면 된다. placeholder의 전달 파라미터는 다음과 같다.
#
# placeholder(
#     dtype,
#     shape=None,
#     name=None
# )
# dtype : 데이터 타입을 의미하며 반드시 적어주어야 한다.
# shape : 입력 데이터의 형태를 의미한다. 상수 값이 될 수도 있고 다차원 배열의 정보가 들어올 수도 있다. ( 디폴트 파라미터로 None 지정 )
# name : 해당 placeholder의 이름을 부여하는 것으로 적지 않아도 된다.  ( 디폴트 파라미터로 None 지정 )

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))

# 아래는 웹 예제
# 위에서도 말했었지만, placeholder는 다른 텐서를 할당하는 것이라고 했다.
# 이를 할당하기 위해서는 feed dictionary 라는 것을 활용하게 되는데 세션을 생성할때 feed_dict의 키워드 형태로 텐서를 맵핑 할 수 있다.
# 위와 같이 선언 후 feed_dict 변수를 할당해도 되고 바로 값을 대입시켜도 무방하다.

p_holder1 = tf.placeholder(dtype=tf.float32)
p_holder2 = tf.placeholder(dtype=tf.float32)
p_holder3 = tf.placeholder(dtype=tf.float32)

val1 = 5
val2 = 10
val3 = 3

ret_val = p_holder1 * p_holder2 + p_holder3

feed_dict = {p_holder1: val1, p_holder2: val2, p_holder3: val3}
result = sess.run(ret_val, feed_dict=feed_dict)

print(result)


print("4. ##########################################################################")