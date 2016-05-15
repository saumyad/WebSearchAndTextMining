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
scrptPth = os.path.join(os.path.dirname(__file__), './TestImages')
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

			if filename.endswith((".JPEG", ".jpeg")):# and count < 10:
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
	# std_scale = preprocessing.StandardScaler().fit(ImageData)
	# ImageData = std_scale.transform(ImageData)
	print "Pre Processed ", count, " images"
	return ImageData


def categorize_images(vectorImages):
	global finalVectorImages 
	for i in xrange(len(vectorImages)):
		finalVectorImages.append([vectorImages[i], realCategory[i]])

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
	
	return accuracy
	

def main():
	iterloops = 1
	# collect image data
	inputData = getInputData()
			
	#Q2_1: Auto-encoders
	vectorImages = np.linalg.svd(inputData, full_matrices=True)
	#print vectorImages[0]#, vectorImages[3], vectorImages[5]
	categorize_images(vectorImages[0])
	print "Finding 5 nearest neighbors through Euclidean Distance..."
	#Q4_1: KNN through euclidean
	avg_accuracy_eucl = 0
	for x in range(iterloops):
		accuracy_eucl = find_5NN('e')
		avg_accuracy_eucl += accuracy_eucl
	print('Average Accuracy of Euclidean: ' + repr((avg_accuracy_eucl/iterloops)*100) + '%')	
	print
	
	print "Finding 5 nearest neighbors through Pearson Correlation Coefficient..."
	#Q4_1: KNN through pearson
	avg_accuracy_pear = 0
	for x in range(iterloops):
		accuracy_pear = find_5NN('p')
		avg_accuracy_pear += accuracy_pear
	
	print('Average Accuracy of Euclidean: ' + repr((avg_accuracy_pear/iterloops)*100) + '%')	
	

#Q4_3: Confusion matrix

#Q2_2: SVD
#q2_2 = np.linalg.svd(inputData, full_matrices=True)
#print q2_2

if __name__ == "__main__":
	main()
