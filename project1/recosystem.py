#N=1000 users
#Find some way to maintain mapping between img# and category-->dictionary!
#Use a random generator for numpy ndarray
#blah!

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import os, os.path, sys
import fnmatch
import tensorflow as tf
import math
import random
import scipy.stats
from sklearn import preprocessing
import operator
from sklearn.cross_validation import train_test_split
from scipy.stats.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 

ht_new = 100#256
w_new = 100#256

categorymap = {'Animal':343, 'Fungus':1207, 'Geological':1808, 'Person':1242, 'PlantFloraPlantLife':1271, 'Sport':1888, 'Dummy':0}
categorymap_cumulat_start = {'Animal':0, 'Fungus':343, 'Geological':1550, 'Person':3358, 'PlantFloraPlantLife':4600, 'Sport':5871, 'Dummy':7759}

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
	print 'found neighbours', neighbors
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
	print 'got response', sortedVotes
	return int(sortedVotes[0][0])

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1
	print 'got accuracy'
	return (correct/float(len(testSet))) * 100.0

def splitData(finalVectorImages, training_data=[] , testing_data=[]):
	#generates a different testing training set every time
	# 3/4th training, 1/4th testing
	training_data, testing_data = train_test_split(finalVectorImages, test_size=0.20)
	return training_data, testing_data

def getConfusionMatrix(y_true, y_pred):
	print "Confusion Matrix is ", confusion_matrix(y_true, y_pred)

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
		print('-- Predicted category = ' + reverse_image_category[result] + ', actual = ' + reverse_image_category[testing_data[x][-1]])
	getConfusionMatrix(realValues, predictions)
	accuracy = getAccuracy(testing_data, predictions)
	print('Accuracy: ' + repr(accuracy*100) + '%')

def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(xrange(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]
			
def getImage_RGB(path):
	image = Image.open(path)
	arr = np.asarray(image)
	return image, arr

def getInputData_BW():
	ImageData = np.ndarray(shape=(1, (w_new * ht_new)))
	scrptPth = '/home/saniya/Documents/WSTM/Project1/Data/Person'
	#os.path.dirname(os.path.realpath(__file__))				
	
	print 'here'
	for root, dirs, files in os.walk(scrptPth):
	   	print root, dirs
	   	for filename in files:
			if filename.endswith((".JPEG", ".jpeg")):
				#filename = os.path.join(root, filename)
				image_path = root + '/' + filename
				image_rgb, arr_rgb = getImage_RGB(image_path)
				image_bw = image_rgb.convert("L")
				arr_bw = np.asarray(image_bw)

				image_resize = image_bw.resize((w_new, ht_new), Image.ANTIALIAS)
				arr_resize = np.asarray(image_resize)
				arr_flat = arr_resize.flatten()

				ImageData = np.vstack((ImageData, arr_flat))
	print "data", ImageData.shape
	return ImageData

def categorize_images(vectorImages, realCategory):
	finalVectorImages = []
	for i in xrange(len(vectorImages)):
		finalVectorImages.append([vectorImages[i], realCategory[i]])
	return finalVectorImages

N_users = 50#00
N_images = 7759

rating_matrix = np.ndarray(shape=(N_users, N_images), dtype=float)
categorymap = {'Animal':343, 'Fungus':1207, 'Geological Formation':1808, 'Person':1242, 'Plant, flora, plant life':1271, 'Sport':1888, 'Dummy':0}
categorymap_cumulat_start = {'Animal':0, 'Fungus':343, 'Geological Formation':1550, 'Person':3358, 'Plant, flora, plant life':4600, 'Sport':5871, 'Dummy':7759}

image_category_index = {'Animal':1, 'Fungus':2, 'Geological Formation':3, 'Person':4, 'Plant, flora, plant life':5, 'Sport':6, 'Dummy':7}
reverse_image_category = {1:'Animal', 2:'Fungus', 3:'Geological Formation', 4:'Person', 5:'Plant, flora, plant life', 6:'Sport', 7:'Dummy'}

realCategory = []
i=0
for i in range(N_images):
	if i<categorymap_cumulat_start['Fungus']:
		realCategory.append(1)
	else:
		if i<categorymap_cumulat_start['Geological Formation']:
			realCategory.append(2)
		else:
			if i<categorymap_cumulat_start['Person']:
				realCategory.append(3)
			else:
				if i<categorymap_cumulat_start['Plant, flora, plant life']:
					realCategory.append(4)
				else:
					if i<categorymap_cumulat_start['Sport']:
						realCategory.append(5)
					else:
						realCategory.append(6)


def getCategoryName(num):
	if num==0:
		return 'Animal'
	if num==1:
		return 'Fungus'
	if num==2:
		return 'Geological Formation'
	if num==3:
		return 'Person'
	if num==4:
		return 'Plant, flora, plant life'
	if num==5:
		return 'Sport'
	else:
		return 'Dummy'

M= 500
catMean_low = 1.5
catMean_high = 4.5
catSigma = 1.0
rating_low = 1
rating_high = 5

#for each user
for u in np.arange(N_users):
	#Select 4 categories (geometric)
	cat = random.sample(range(0,6), 4)
	#Select total M movies across the 4 chosen categories: 
	num_imgpercat = constrained_sum_sample_pos(4, M)
	#check if number of img required for a category is lower than max in the category
	k=0
	for i in cat:
		thisCat = getCategoryName(i)
		nextCat = getCategoryName(i+1)
		if (categorymap[thisCat] > num_imgpercat[k]):
			low = categorymap_cumulat_start[thisCat]
			high = categorymap_cumulat_start[nextCat]
			#generate m_i random numbers between low and high
			movieNum = random.sample(range(low, high), num_imgpercat[k])
			catMean = random.uniform(catMean_low, catMean_high)
			a, b = (rating_low - catMean) / catSigma, (rating_high - catMean) / catSigma
			rating_distr = scipy.stats.truncnorm(a,b,scale=catSigma)
			ratingList = rating_distr.rvs(size=num_imgpercat[k])
			ratingList = ratingList * catSigma + catMean
			r=0
			for m in movieNum:
				rating_matrix[u][m] = ratingList[r]
				r+=1
			k+=1
			
print 'rating_matrix', rating_matrix
rm_trans = np.transpose(rating_matrix, (1, 0))
print rm_trans.shape
q3_4, S, V = np.linalg.svd(rm_trans, full_matrices=True)
print "q3_4 done"
print q3_4.shape
q3_4_final = categorize_images(q3_4, realCategory)
print q3_4_final
find_5NN(q3_4_final, realCategory, 0)
find_5NN(q3_4_final, realCategory, 1)

			







