import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph() #reset the graph

# data 가져오기 
f1 = open ('traindata.txt','r')
Xr =[]
Xr =[ line.split()for line in f1]
f1.close()
Xr=np.array(Xr, dtype=np.float32)

f1 = open ('trainlabels.txt','r')
yr =[]
yr =[ line.split()for line in f1]
f1.close()
yr=np.array(yr, np.int64)

f1 = open ('testdata.txt','r')
Xt =[]
Xt =[ line.split()for line in f1]
f1.close()
Xt=np.array(Xt, dtype=np.float32)

f1 = open ('testlabels.txt','r')
yt =[]
yt =[ line.split()for line in f1]
f1.close()
yt=np.array(yt, dtype=np.int64)

sample_size=Xr.shape[0]; samplet_size=Xt.shape[0]; 
Yr=np.zeros((sample_size,10)); Yt=np.zeros((samplet_size,10))
for ii in range(sample_size):
    Yr[ii, yr[ii]]=1
    
for ii in range(samplet_size):
    Yt[ii, yt[ii]]=1      

tf.set_random_seed(1)#the graph-level random seed
learning_rate=0.01
training_epochs=10
batch_size=100

keep_prob=tf.placeholder(tf.float32) # =1-dropout prob
X=tf.placeholder(tf.float32,[None,784])
X_img=tf.reshape(X,[-1,28,28,1]) # = [batch,ht,wt,channels]
Y=tf.placeholder(tf.float32,[None,10])
K1=tf.Variable(tf.random_normal([5,5,1,30],stddev=0.01))
a1=tf.nn.conv2d(X_img, K1, strides=[1,1,1,1], padding='VALID')
#a1=tf.layers.batch_normalization(a1, training=True)# batch normalization
a1=tf.nn.relu(a1) 
a1=tf.nn.dropout(a1, keep_prob)
#a1.shape=(None,24,24,30)
h1=tf.nn.max_pool(a1,ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
h1.shape#=(None,12,12,30)
Flat=tf.reshape(h1,[-1,12*12*30])
W1=tf.get_variable("W1",shape=[12*12*30,10],initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.random_normal([10]))
pred=tf.matmul(Flat, W1)+b1 # or pred=tf.add(tf.matmul(L1_flat, W2), b)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optim=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#tf.train.AdamOptimizer()
#tf.train.RMSPropOptimizer()

correct_predict=tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_predict, tf.float32))

sess=tf.Session(); sess.run(tf.global_variables_initializer())

# Train CNN
print('Learning started')
for epoch in range(training_epochs):
   avg_cost=0
   total_batch=int(sample_size/batch_size)
   for i in range(total_batch):
       batch_xs=Xr[i*batch_size:(i+1)*batch_size]
       batch_ys=Yr[i*batch_size:(i+1)*batch_size]
       feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.9}
       sess.run(optim, feed_dict=feed_dict)
       ccost=sess.run(cost, feed_dict=feed_dict)
       avg_cost+=ccost/total_batch
       acc=sess.run(accuracy, feed_dict=feed_dict)
   print('Epoch:' '%04d' %(epoch+1),'cost=','{:.6f}'.format(avg_cost),'accuracy=','{:.4f}'.format(acc) )
print('Learning finished')

# Test CNN with training data and test data
print('Accuracy(tr):', sess.run(accuracy, feed_dict={X:Xr, Y: Yr,keep_prob:0.9}))
print('Accuracy(ts):', sess.run(accuracy, feed_dict={X:Xt, Y: Yt,keep_prob:0.9}))

# Test data 그림
plt.imshow(Xt[1].reshape(28,28),cmap='Greys')
