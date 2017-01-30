#!/usr/bin/env python
"""A very simple cat vs dog classifier.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import collections
import os
import datetime

import numpy as np
import PIL
from PIL import Image
import tensorflow as tf


TrainTestDataset = collections.namedtuple('TrainTestDataset',['train', 'test'])

class Dataset():
  """Dataset with next_batch capability"""
  def __init__(self, images, labels):
    assert(len(labels) == len(images))
    self.n = len(labels)
    self.images = images
    self.labels = labels

  def next_batch(self, n=100):
    "Return a random subset of n images and labels"""
    n = min(self.n,n)
    if n == 0:
      return (np.zeros([0,64*64]), np.zeros([0]))
    indices = np.random.choice(self.n, n, replace=False)
    return (self.images[indices], self.labels[indices])
    

def countPNGs(path):
  return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and name.endswith('png')])


def load_cats_dogs(cat_dir, dog_dir, train_test_ratio=0.75):
  print("Loading %s and %s" % (cat_dir, dog_dir))
  cat_n = countPNGs(cat_dir)
  dog_n = countPNGs(dog_dir)
  print('  %s contains %d files' % (cat_dir, cat_n))
  print('  %s contains %d files' % (dog_dir, dog_n))

  # Labels are one-hot, 2 columns, first is cat second is dog
  labels = np.zeros([cat_n+dog_n, 2], dtype=np.float32)
  labels[:cat_n,0] = 1
  labels[cat_n:,1] = 1

  # Images are flattened n x 64*64 of the grayscale pixel values as float32s
  flat_images = np.zeros([cat_n+dog_n, 64*64], dtype=np.float32)

  # Load images into flat_images
  # Add cats
  for i in range(cat_n):
    filepath = os.path.join(cat_dir, "%04d.png" % i)
    flat_images[i,:] = np.array(Image.open(filepath)).flatten()

  # Add dogs
  for i in range(dog_n):
    filepath = os.path.join(dog_dir, "%04d.png" % i)
    flat_images[cat_n + i,:] = np.array(Image.open(filepath)).flatten()

  return splitIntoTrainingAndTestDatasets(flat_images, labels, train_test_ratio)

def splitIntoTrainingAndTestDatasets(flat_images, labels, ratio=0.7):
  n = len(labels)
  cutoff = int(n*ratio)
  if cutoff == 0 or cutoff == n:
    raise Exception('Not enough data to split into training/test')

  # First shuffle images and labels
  new_order = np.random.permutation(np.arange(n))
  flat_images = flat_images[new_order]
  labels = labels[new_order]

  training = Dataset(flat_images[:cutoff], labels[:cutoff])
  test = Dataset(flat_images[cutoff:], labels[cutoff:])
  return TrainTestDataset(train=training, test=test)


FLAGS = None

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def main(_):
  # Import data
  cat_dog_dataset = load_cats_dogs(FLAGS.cat_dir, FLAGS.dog_dir)
  print("Contains %d Training samples" % cat_dog_dataset.train.n)
  print("Contains %d Test samples" % cat_dog_dataset.test.n)

  sess = tf.InteractiveSession()

  # Create the model
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 64*64], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
  
  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 64, 64, 1])
    tf.summary.image('input', image_shaped_input, 10)

  hidden1 = nn_layer(x, 64*64, 20, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 20, 2, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = cat_dog_dataset.train.next_batch(100)
      k = FLAGS.dropout
    else:
      xs, ys = cat_dog_dataset.test.images, cat_dog_dataset.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cat_dir', type=str, default='images/cats',
                      help='Directory for storing input cat images')
  parser.add_argument('--dog_dir', type=str, default='images/dogs',
                      help='Directory for storing input dog images')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  output_dir = '/tmp/tensorflow/catdog/' + datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S") + '/'
  print('Default log output dir: %s' % output_dir)
  parser.add_argument('--log_dir', type=str, default=output_dir,
                      help='Summaries log directory')
  print('Log output dir used: %s' % FLAGS.output_dir)
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
