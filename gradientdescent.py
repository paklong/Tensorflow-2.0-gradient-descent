import tensorflow as tf

#Function with inputs x1 and x2
def fu(x1, x2): 
	return x1 ** 2.0 - x1 * 3  + x2 ** 2

#Function without input
def fu_minimzie():
	return x1 ** 2.0 - x1 * 3  + x2 ** 2

#Reset the values of x1 and x2 for different algorithms
def reset():	
	x1 = tf.Variable(10.0) 
	x2 = tf.Variable(10.0) 
	return x1, x2

#Way1: without optimizers, pure math computation
x1, x2 = reset()
for i in range(50):	
	#Find partial derivatives of x1 and x2 with respect to y using auto differentiation
	with tf.GradientTape() as tape:
		y = fu(x1, x2)
	grads = tape.gradient(y, [x1, x2])
	print ('y = {:.1f}, x1 = {:.1f}, x2 = {:.1f},  grads0 = {:.1f}, grads1 = {:.1f} '.format(y.numpy(), x1.numpy(), x2.numpy(), grads[0].numpy(), grads[1].numpy()))
	#Update x1, x2
	x1.assign(x1 - 0.1*grads[0].numpy())
	x2.assign(x2 - 0.1*grads[1].numpy())

#Way2: with optimizers biut without minimize funciton
x1, x2 = reset()
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
for i in range(50):
	#Find partial derivatives of x1 and x2 with respect to y using auto differentiation
	with tf.GradientTape() as tape:
		y = fu(x1, x2)
	grads = tape.gradient(y, [x1, x2])
	#Process the gradients
	processed_grads = [g for g in grads]
	grads_and_vars = zip(processed_grads, [x1, x2])
	print ('y = {:.1f}, x1 = {:.1f}, x2 = {:.1f},  grads0 = {:.1f}, grads1 = {:.1f} '.format(y.numpy(), x1.numpy(), x2.numpy(), grads[0].numpy(), grads[1].numpy()))
	#Update x1, x2
	opt.apply_gradients(grads_and_vars)

#Way3: using minimize funciton
x1, x2 = reset()
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
for i in range(50):
	print ('y = {:.1f}, x1 = {:.1f}, x2 = {:.1f}'.format(fu(x1, x2).numpy(), x1.numpy(), x2.numpy()))
	#Update x1 an x2
	#Find partial derivatives of x1 and x2 with respect to y using auto differentiation and Update x1, x2
	opt.minimize(fu_minimzie, var_list=[x1, x2])
	# print (opt.get_gradients(fu(x1, x2), [x1, x2]))
