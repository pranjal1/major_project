#! /usr/bin/env python

# TYO PARAMETERS HARU ADJUST GARNA MATRA BAAKI HO...... like image size in this script and on model_alexNet.py 

import pickle_load
import model_alexNet
import tensorflow as tf
import img_proc

data = pickle_load.Dataset('/home/aashish/Documents/cifar-10-batches-py')

# Parameters
learn_rate = 0.001
decay_rate = 0.1
batch_size = 64
display_step = 20

#n_classes = data.total_data # we got mad kanji
n_classes = 10
dropout = 0.8 # Dropout, probability to keep units
imagesize = 24
img_channel = 3
inference = model_alexNet.modelAlexNet()
x = tf.placeholder(tf.float32, [None, imagesize, imagesize, img_channel])
#distorted_images = img_proc.pre_process(images=x, training=True)
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

pred = inference.model_predict(x, keep_prob, n_classes, imagesize, img_channel)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

global_step = tf.Variable(initial_value=0,name = 'global_step', trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, 1000, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred)
tf.add_to_collection("accuracy", accuracy)

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < 3000:
        batch_xs, batch_ys, batch_yhot = data.getNextBatch(batch_size)
        batch_xs = img_proc.pre_process(batch_xs).eval()
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_yhot, keep_prob: dropout})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_yhot, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_yhot, keep_prob: 1.})
            rate = sess.run(lr)
            print "lr " + str(rate) + " Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

        if step % 1000 == 0:
            saver.save(sess, 'save/model.ckpt', global_step=step*batch_size)
        step += 1
    print "Optimization Finished!"
    step_test = 1
    while step_test * batch_size < len(testing):
        testing_ys, testing_xs = testing.nextBatch(batch_size)
        print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1.})
        step_test += 1
