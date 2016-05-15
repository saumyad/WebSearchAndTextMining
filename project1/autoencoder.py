import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path, sys
import fnmatch
import tensorflow as tf
import math
import random
from sklearn import preprocessing
import operator
from sklearn.cross_validation import train_test_split
from scipy.stats.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

ht_new = 100#256
w_new = 100#256
count = 0
scrptPth = os.path.join(os.path.dirname(__file__), './Images')
finalVectorImages = []
# store the real category the images belong to
realCategory = []
image_category_index = {'Animal':1, 'Fungus':2, 'Geological Formation':3, 'Person':4, 'Plant, flora, plant life':5, 'Sport':6, 'Dummy':7}
reverse_image_category = {1:'Animal', 2:'Fungus', 3:'Geological Formation', 4:'Person', 5:'Plant, flora, plant life', 6:'Sport', 7:'Dummy'}


def euclideanDistance(vector1, vector2):
	## much faster to use vectors
	return np.sum((vector1-vector2) ** 2)
	
def getEuclideanNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance[0], trainingSet[x][0])
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def pearsonCorrelation(vector1, vector2):
	## much faster to use vectors
	return pearsonr(vector1, vector2)[0]
	
def getPearsonNeighbors(trainingSet, testInstance, k):
	distances = []
	for x in range(len(trainingSet)):
		coeff = pearsonCorrelation(testInstance[0], trainingSet[x][0])
		distances.append((trainingSet[x], coeff))
	distances.sort(key=operator.itemgetter(1), reverse=True)
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(y_true, y_pred):
	return accuracy_score(y_true, y_pred)

def getInputData():
	ImageData = np.ones(shape = (0,w_new * ht_new))
	tempArr = []
	global count
	for root, dirs, files in os.walk(scrptPth):
	   	
	   	for filename in files:

			if filename.endswith((".JPEG", ".jpeg")):# and count < 80:
				image_category = root.split('/')[-1]
				image_path = os.path.join(root, filename)
				image = Image.open(image_path)
				image_bw = image.convert("L")
				arr_bw = np.asarray(image_bw)
				image_resize = image_bw.resize((w_new, ht_new), Image.ANTIALIAS)
				arr_resize = np.asarray(image_resize)
				
				arr_flat = arr_resize.flatten()				
				#normalize the image vector
				arr_flat = arr_flat * (255.0/arr_flat.max())
				tempArr.append(arr_flat)
				#find category
				if image_category in image_category_index:
					realCategory.append(image_category_index[image_category])
				else:
					print "not in dict ", image_category
					realCategory.append(7)
				count += 1
	
	if (not tempArr):
		print "No image information obtained from file traversal"
		return -1
	ImageData = np.vstack(tempArr)
	del tempArr
	#normalize the feature vector across images
	std_scale = preprocessing.StandardScaler().fit(ImageData)
	ImageData = std_scale.transform(ImageData)
	print "Pre Processed ", count, " images"
	return ImageData

def create(x, layer_sizes):

	# Build the encoding layers
	next_layer_input = x

	encoding_matrices = []
	for dim in layer_sizes:
	#	print next_layer_input
		input_dim = int(next_layer_input.get_shape()[1])
		#input_dim = int(next_layer_input.shape[1])

		# Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
		# W has a shape of [input_dim, dim] because we want to multiply the n-dimensional 
		# image vectors by it to produce layersize-dimensional vectors of evidence for the difference classes.
		W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim), dtype=tf.float32))

		# Initialize b to zero
		# bias added to the weights
		b = tf.Variable(tf.zeros([dim], dtype=tf.float32))

		# We are going to use tied-weights so store the W matrix for later reference.
		encoding_matrices.append(W)

		# constrain the neurons to be inactive most of the time. This discussion assumes a sigmoid activation function. If you are using a tanh activation function,
		# then we think of a neuron as being inactive when it outputs values close to -1.
		output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)

		# the input into the next layer is the output of this layer
		next_layer_input = output

	# The fully encoded x value is now stored in the next_layer_input
	encoded_x = next_layer_input

	# build the reconstruction layers by reversing the reductions
	layer_sizes.reverse()
	encoding_matrices.reverse()


	for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]) :
		# we are using tied weights, so just lookup the encoding matrix for this step and transpose it
		W = tf.transpose(encoding_matrices[i])
		b = tf.Variable(tf.zeros([dim]))
		output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
		next_layer_input = output

	# the fully encoded and reconstructed value of x is here:
	reconstructed_x = next_layer_input

	return {
		'encoded': encoded_x,
		'decoded': reconstructed_x,
		'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
	}
def categorize_images(vectorImages):
	global finalVectorImages 
	for i in xrange(len(vectorImages)):
		finalVectorImages.append([vectorImages[i], realCategory[i]])

def generateEncodings(inputData):
	global count
	vectorImages = []
	sess = tf.Session()
	start_dim = w_new * ht_new
	x = tf.placeholder("float", [None, start_dim])
	autoencoder = create(x, [1000])  ## how to decide [4,2]
	init = tf.initialize_all_variables()
	print "Tensorflow session started..."
	print ""
	sess.run(init)
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(autoencoder['cost'])
	
	num_training_steps = count/25
	rem_steps = count%25
	# do loads of training steps
	for i in range(num_training_steps):
		# make a batch of 15:
		batch = []
		for j in range(25):
  			batch.append(inputData[j*i+j])
		sess.run(train_step, feed_dict={x: np.array(batch)})
		tmp = sess.run(autoencoder['encoded'], feed_dict={x: batch})
		for p in range(25):
			vectorImages.append(tmp[p])
		if i % 5 == 0:
			print "Cost involved in autoencoding the next batch of images..", sess.run(autoencoder['cost'], feed_dict={x: batch})
	
	batch = []
	#print num_training_steps, count, rem_steps
	finished = num_training_steps*25
	
	for i in range(rem_steps):
		batch.append(inputData[finished+i])
		sess.run(train_step, feed_dict={x: np.array(batch)})
	tmp = sess.run(autoencoder['encoded'], feed_dict={x: batch})
	for p in range(rem_steps):
		vectorImages.append(tmp[p])
	print "Cost involved in autoencoding the last batch of images..", sess.run(autoencoder['cost'], feed_dict={x: batch})
	
	#print "fg", vectorImages[0], vectorImages[1], vectorImages[2]
	categorize_images(vectorImages)	
	print len(finalVectorImages), " Images converted to vectors"
	del vectorImages

def splitData(training_data=[] , testing_data=[]):
	#generates a different testing training set every time
	# 3/4th training, 1/4th testing
	training_data, testing_data = train_test_split(finalVectorImages)
	return training_data, testing_data

def getConfusionMatrix(y_true, y_pred):
	print "-----------Confusion Matrix is------------- "
	print confusion_matrix(y_true, y_pred)


def find_5NN(method):
	# prepare data
	training_data = []
	testing_data = []
	training_data, testing_data = splitData(training_data, testing_data)
	print "Testing ", len(testing_data), " sample images...."
	# generate predictions
	predictions=[]
	realValues = []
	k = 5
	for x in range(len(testing_data)):
		if (method == 'e'):
			neighbors = getEuclideanNeighbors(training_data, testing_data[x], k)
		else:
			neighbors = getPearsonNeighbors(training_data, testing_data[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		realValues.append(testing_data[x][-1])
	#	print('-- Predicted category = ' + reverse_image_category[result] + ', actual = ' + reverse_image_category[testing_data[x][-1]])
	
	getConfusionMatrix(realValues, predictions)
	accuracy = getAccuracy(realValues, predictions)
	print('Accuracy: ' + repr(accuracy*100) + '%')
	

def main():
	# collect image data
	inputData = getInputData()
			
	#Q2_1: Auto-encoders
	generateEncodings(inputData)

	print "Finding 5 nearest neighbors through Euclidean Distance..."
	#Q4_1: KNN through euclidean
	find_5NN('e')

	print
	print "Finding 5 nearest neighbors through Pearson Correlation Coefficient..."
	#Q4_1: KNN through pearson
	find_5NN('p')

#Q4_3: Confusion matrix

#Q2_2: SVD
#q2_2 = np.linalg.svd(inputData, full_matrices=True)
#print q2_2

if __name__ == "__main__":
	main()
