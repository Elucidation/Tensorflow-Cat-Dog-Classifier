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


def load_cats_dogs(cat_dir, dog_dir, train_test_ratio=0.9):
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

def main(_):
  # Import data
  cat_dog_dataset = load_cats_dogs(FLAGS.cat_dir, FLAGS.dog_dir)
  print("Contains %d Training samples" % cat_dog_dataset.train.n)
  print("Contains %d Test samples" % cat_dog_dataset.test.n)

  # print(train.images)
  # a,b = train.next_batch(n=10)
  # print(a.shape, a)
  # print(b.shape, b)


  # Create the model
  x = tf.placeholder(tf.float32, [None, 64*64])
  W = tf.Variable(tf.zeros([64*64, 2]))
  b = tf.Variable(tf.zeros([2]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  # Train
  for i in range(FLAGS.num_steps):
    batch_xs, batch_ys = cat_dog_dataset.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 100 == 0:
      # Do Test step and plot accuracy
      accuracy_score = 100 * sess.run(accuracy, feed_dict={x: cat_dog_dataset.test.images,
                                          y_: cat_dog_dataset.test.labels})
      print("% 4d/%d : Accuracy: %.2f%%" % (i, FLAGS.num_steps, accuracy_score))

  # Test trained model
  accuracy_score = 100 * sess.run(accuracy, feed_dict={x: cat_dog_dataset.test.images,
                                      y_: cat_dog_dataset.test.labels})
  print("Accuracy: %.2f%%" % accuracy_score)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cat_dir', type=str, default='images/cats',
                      help='Directory for storing input cat images')
  parser.add_argument('--dog_dir', type=str, default='images/dogs',
                      help='Directory for storing input dog images')
  parser.add_argument('--num_steps', type=int, default=1000,
                      help='Number of steps to train model')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  # main(None)
