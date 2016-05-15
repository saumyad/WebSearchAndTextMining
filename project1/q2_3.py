import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
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
ht_new_hist = 256
w_new_hist = 256

count = 0
scrptPth = os.path.join(os.path.dirname(__file__), './TestImages')
finalVectorImages = []
# store the real category the images belong to
image_category_index = {'Animal':1, 'Fungus':2, 'Geological Formation':3, 'Person':4, 'Plant, flora, plant life':5, 'Sport':6, 'Dummy':7}
reverse_image_category = {1:'Animal', 2:'Fungus', 3:'Geological Formation', 4:'Person', 5:'Plant, flora, plant life', 6:'Sport', 7:'Dummy'}

def getConfusionMatrix(y_true, y_pred):
	print "Confusion Matrix is ", confusion_matrix(y_true, y_pred)

def getImage_RGB(path):
	image = Image.open(path)
	arr = np.asarray(image)
	return image, arr
			
def getImage_RGB_1(path):
	image = cv2.imread(path, cv2.IMREAD_COLOR)
	arr = np.asarray(image)
	return image, arr

def getFlattenedHistogram_RGB(arr_2_3, plot):
	chans = cv2.split(arr_2_3)
	colors = ("b", "g", "r")
	if(plot == 1):
		fig_2_3 = plt.figure()
		plt.title("RGB Histogram")
		p1 = fig_2_3.add_subplot(111)

	features = []
	i=0
	fillup = np.zeros(256)

	for (chan, color) in zip(chans, colors):
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		i+=1
		features.extend(hist)
		if(plot == 1):
			p1.plot(hist, color=color)
	while(i<3):
		features.extend(fillup)
		i+=1

	return np.array(features).flatten()

def getFlattenedHistogram_HSV(arr_2_3, plot):
	hsv = cv2.cvtColor(arr_2_3, cv2.COLOR_BGR2HSV)
	features = []
	if(plot == 1) :
		fig_2_3 = plt.figure()
		plt.title("HSV Histogram")
		p2 = fig_2_3.add_subplot(111)
	hist = cv2.calcHist([hsv], [0,1], None, [180, 256], [0, 180, 0, 256])
	if(plot == 1) :
		p2.plot(hist)
	features.extend(hist)
	remaining = 46080 - len(np.array(features).flatten())
	if(remaining > 0):
		features.extend(np.zeros(remaining))
	return np.array(features).flatten()

def euclideanDistance(vector1, vector2):
	## much faster to use vectors
	return np.sum((vector1-vector2) ** 2)

def pearsonCorrelation(vector1, vector2):
	## much faster to use vectors
	return pearsonr(vector1, vector2)[0]
	
def getNeighbors(trainingSet, testInstance, k, isEuclidean):
	distances = []
	for x in range(len(trainingSet)):
		if(isEuclidean == 1) :
			#coeff = euclideanDistance(testInstance[0], trainingSet[x][0])
			coeff = euclideanDistance(testInstance[0], trainingSet[x][0])
		else:
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
	return int(sortedVotes[0][0])

def getAccuracy(y_true, y_pred):
	return accuracy_score(y_true, y_pred)

def splitData(finalVectorImages, training_data=[] , testing_data=[]):
	#generates a different testing training set every time
	# 3/4th training, 1/4th testing
	training_data, testing_data = train_test_split(finalVectorImages, test_size=0.05)
	return training_data, testing_data

def find_5NN(finalVectorImages, realCategory, isEuclidean):
	# prepare data
	training_data = []
	testing_data = []
	training_data, testing_data = splitData(finalVectorImages, training_data, testing_data)
	print "Testing ", len(testing_data), " sample images...."
	# generate predictions
	predictions=[]
	realValues = []
	k = 5
	for x in range(len(testing_data)):
		neighbors = getNeighbors(training_data, testing_data[x], k, isEuclidean)
		result = getResponse(neighbors)
		predictions.append(result)
		realValues.append(testing_data[x][-1])
		print('-- Predicted category = ' + reverse_image_category[result] + ', actual = ' + reverse_image_category[testing_data[0][-1]])
	getConfusionMatrix(realValues, predictions)
	accuracy = getAccuracy(realValues, predictions)
	print('Accuracy: ' + repr(accuracy*100) + '%')
			
def getImage_RGB(path):
	image = Image.open(path)
	arr = np.asarray(image)
	return image, arr

def getInputData_RGBHistogram():
	ImageData = np.ndarray(shape=(0, 768))
	#scrptPth = '/home/saniya/Documents/WSTM/Project1/TestImages'
	realCat_RGB = []
	for root, dirs, files in os.walk(scrptPth):
	   	for filename in files: 
			if filename.endswith((".JPEG", ".jpeg")):
				image_category = root.split('/')[-1]
				image_path = root + '/' + filename
				image_rgb, arr_rgb = getImage_RGB_1(image_path)
				image_resize = cv2.resize(image_rgb, (w_new_hist, ht_new_hist))
				arr_resize = np.asarray(image_resize)
				arr_flat = getFlattenedHistogram_RGB(arr_resize, 0)				
				ImageData = np.vstack((ImageData, arr_flat))
				#find category
				if image_category in image_category_index:
					realCat_RGB.append(image_category_index[image_category])
				else:
					print "not in dict ", image_category
					realCat_RGB.append(7)
	return ImageData, realCat_RGB

def getInputData_HSVHistogram():
	ImageData = np.ndarray(shape=(0, 46080))
	#scrptPth = '/home/saniya/Documents/WSTM/Project1/TestImages'
	realCat_HSV = []
	for root, dirs, files in os.walk(scrptPth):
	   	for filename in files:
			if filename.endswith((".JPEG", ".jpeg")):
				image_category = root.split('/')[-1]
				image_path = root + '/' + filename
				image_rgb, arr_rgb = getImage_RGB_1(image_path)
				image_resize = cv2.resize(image_rgb, (w_new_hist, ht_new_hist))
				arr_resize = np.asarray(image_resize)
				arr_flat = getFlattenedHistogram_HSV(arr_resize, 0)
				ImageData = np.vstack((ImageData, arr_flat))
				#find category
				if image_category in image_category_index:
					realCat_HSV.append(image_category_index[image_category])
				else:
					print "not in dict ", image_category
					realCat_HSV.append(7)
	return ImageData, realCat_HSV

def categorize_images(vectorImages, realCategory):
	finalVectorImages = []
	for i in xrange(len(vectorImages)):
		finalVectorImages.append([vectorImages[i], realCategory[i]])
	return finalVectorImages

#Q2_3_1
q2_3_1, realCat_RGB = getInputData_RGBHistogram()
q2_3_1_final = categorize_images(q2_3_1, realCat_RGB)
find_5NN(q2_3_1_final, realCat_RGB, 1)
print "q2_3_1 done"

q2_3_2, realCat_HSV = getInputData_HSVHistogram()
q2_3_2_final = categorize_images(q2_3_2, realCat_HSV)
find_5NN(q2_3_2_final, realCat_HSV, 1)
print "q2_3_2 done"

#Sample Histograms
path_2_3 = '/home/saniya/Documents/WSTM/Project1/TestImages/Animal/n00015388_355.JPEG'
img_2_3, arr_2_3 = getImage_RGB(path_2_3)
blah = getFlattenedHistogram_RGB(arr_2_3, 1)
blah = getFlattenedHistogram_HSV(arr_2_3, 1)

plt.show()


