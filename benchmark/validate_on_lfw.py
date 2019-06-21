"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import tensorflow as tf
import numpy as np
import argparse
#import facenet
#import lfw
import os
import sys
from PIL import Image
import ast
#from tensorflow.python.ops import data_flow_ops

"""
# scipy module cannot be surpported
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
"""

from custom.facenet_tpu import FacenetEngine
from benchmark import lfw
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16

def main(args):
  
    # Read the file containing the pairs used for testing
    pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
    
    #image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
    #labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
    batch_size = args.lfw_batch_size
    #control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
    #phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    nrof_preprocess_threads = 4
    image_size = (args.image_size, args.image_size)

    # Load the model
    facenet = FacenetEngine(args.model, None)

    # Get output tensor
    #embedding is the list of embedding vector for training images
    #TODO: change the dummy vector into real embedding
    #embeddings = [facenet.GetEmbeddingVector(Image.open(path)) for path in paths]              
    embeddings=None
    if args.emb_file:
        emb_file = open("emb_lfw_mtcnnalign_160.txt", "a")
        emb_file.write("[")
        for i, path in enumerate(paths):
            inf_time,_,emb = facenet.GetEmbeddingVector(Image.open(path))
            emb_file.write(str(emb)+",\n")
            print("time, index: ",str(inf_time)+"ms", str(i))
        emb_file.write("]")
        emb_file.close()
        emb_file = open("emb_lfw_mtcnnalign_160.txt", "r")
        embeddings = ast.literal_eval(emb_file.read())
        emb_file.close()
    else:
        embeddings = []
        for i, path in enumerate(paths):
            inf_time,_,emb = facenet.GetEmbeddingVector(Image.open(path))
            embeddings.append(emb)
            print("time, index: ",str(inf_time)+"ms", str(i))
    #embeddings = [np.zeros(128, ) for _ in paths]              
    evaluate(facenet, embeddings, paths, actual_issame, batch_size, args.lfw_nrof_folds, args.distance_metric, args.subtract_mean, args.use_flipped_images, args.use_fixed_image_standardization)

              
def evaluate(engine, embeddings, image_paths, actual_issame, batch_size, nrof_folds, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization):
    """
    #TODO:cannot work Flip mode now
    """
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int64)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*FLIP
    
    ##sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(engine.get_all_output_tensors_sizes()[0]) # 128 or 512
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images, ))
    for i in range(nrof_batches):
        ##feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = np.asarray(embeddings[i*batch_size:(i+1)*batch_size]), np.asarray(labels_array[i*batch_size:(i+1)*batch_size])
        
        lab_array[lab] = lab
        emb_array[lab, :] = emb.reshape([emb.shape[0], 1, emb.shape[1]])
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    """
    Using scipy: cannot be supported

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
    """

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
        help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--emb_file', 
        help='', type=bool)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
