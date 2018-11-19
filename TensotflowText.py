import tensorflow as tf
import numpy as np
import matplotlib.pylot as plt
np.random.seed(5)
steps=3000
learning_rate=0.01
x_data=np.linspace(-1,1,100)[,np.newaxis]
y_data=np.squard(x_data)*0.4+np.random.randn(*x_data.shape)*0.5
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
weight_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros[1,10])
Output_L1=tf.matmul(x,weight_L1)+biases_L1
L1=tf.nn.tanh(Output_L1)
weight_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros[1,1])
Output_L2=tf.matmul(L1,weight_L2)+biases_L2
pred=tf.nn.tanh(Output_L2)
loss=tf.reduce_mean(tf.square(y-pred))
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
plt.figure()
plt.scatter(x_data,y_data)
with tf.Session() as sess:
  init=tf.global_variables_initializer()
  sess.run(init)
  for i in range(steps):
    sess.run(train,feed_dict={x:x_data,y:y_data})
  pred_value=sess.run(pred,feed_dict={x:x_data})
  plt.plot(x_data,pred_value)
  plt.show()
