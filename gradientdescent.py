import tensorflow as tf


def fu(x):
	return x ** 2.0 - x * 3 
	
x = tf.Variable(10.0) 
for i in range(50):	
	with tf.GradientTape() as tape:
		y = fu(x)
	grads = tape.gradient(y, x)
	print ('y = {:.1f}, x = {:.1f}, grads = {:.1f}'.format(y.numpy(), x.numpy(), grads.numpy()))
	x.assign(x - 0.1*grads.numpy())
