import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import random
import math
import operator

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance+= pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def loadDataet(filename,split,trainingSet=[],testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset= list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]= float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def get_sorted_neighbors(trainingSet,testInstance):
    distances=[]
    length= len(testInstance)-1
    for x in range(len(trainingSet)):
        dist= euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key= operator.itemgetter(1)) #sorts by the dist
    neighbors= []
    for x in range(len(distances)):
        neighbors.append(distances[x][0])
    return neighbors

def get_K_Neighbors(trainingSet,testInstance,k):
    distances=[]
    length= len(testInstance)-1
    for x in range(len(trainingSet)):
        dist= euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key= operator.itemgetter(1)) #sorts by the dist
    neighbors= []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes= {} #dict
    for x in range(len(neighbors)):
        response = neighbors[x][-1] #the class
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortesVotes= sorted(classVotes.iteritems(), key= operator.itemgetter(1),reverse=True)
    return sortesVotes[0][0]

def getAccuracy(testSet,prediction):
    correct= 0
    for x in range(len(testSet)):
        # if testSet[x][-1] is prediction[x]:
        #print "x= "+ "%s"%x
        if testSet[x][-1] == prediction[x]:
            correct+=1
    # return (correct/float(len(testSet)))*100.0
    return (float)("%.2f" % round(((correct / float(len(testSet))) * 100.0),2))

def import_iris_dataset():
    with open('/Users/michalkeren/Desktop/iris.data','rb') as csvfile:
        lines= csv.reader(csvfile)
        for row in lines:
            print ', '.join(row)


def get_sampled_Gaussian(mean,sd,samples_num):
    points = np.round(np.random.normal(mean,sd,samples_num))
    return points

def disply_hist(data):
    _ = plt.hist(data, bins=100)
    _ = plt.xlabel('x-val')
    _ = plt.ylabel('how many points?')
    plt.show()


def disply_bee_swarm(data):
    _ = sns.swarmplot(x='class',y='X', data= data)
    _ = plt.xlabel('class')
    _ = plt.ylabel('X')
    plt.show()
    return


#import_iris_dataset()

# trainingSet=[]
# testSet=[]
# loadDataet('iris.data',0.66,trainingSet,testSet)
# print('Train: ' + repr(len(trainingSet)))
# print('Test: ' + repr(len(testSet)))

# testSet= [[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
# predictions= ['a','a','a']
# accuracy= getAccuracy(testSet,predictions)
# print (accuracy)

# def get_classes_names(data,class_names):
#     for x in range(len(data)):
#         class_name = trainingSet[x][-1]
#         if class_name in class_counter:
#             class_counter[class_name] += 1


def get_training_noise_matrix(trainingSet,predictions,class_names,withPrinting=0):
    #class_names=[]
    class_counter = {}  # dict
    data_dic={}
    for x in range(len(trainingSet)):
        class_name = trainingSet[x][-1]  # the class
        if class_name in class_counter:
            class_counter[class_name] += 1
        else:
            class_counter[class_name] = 1
            class_names.append(class_name)
        key = '%s' % trainingSet[x][-1] +'|%s'% predictions[x]
        if key in data_dic:
            data_dic[key] += 1
        else:
            data_dic[key] = 1
    n = len(class_names)
    data= np.zeros((n, n))
    for i in range(n): #row
        class_tot= class_counter[class_names[i]]
        for j in range(n): #col
            key='%s'%class_names[i] +'|%s'%class_names[j]
            if key in data_dic:
                data[i][j]=float("{0:.3f}".format(data_dic[key]/float(class_tot)))
    names = [_ for _ in class_names]
    if withPrinting:
        noise_df = pd.DataFrame(data, index=names, columns=names)
        print noise_df
    return data

    #print df[1][1]
    #df.to_csv('df.csv', index=True, header=True, sep=' ')

    # trainingSet_copy= trainingSet
    # predictions_copy= predictions
    # for x in range(len(trainingSet)):
    #     if trainingSet_copy[x][-1] == predictions_copy[x]:
    #         trainingSet_copy.pop(x)
    #         predictions_copy.pop(x)

def get_predictions(trainingSet,testSet,k,k_Neg=0):
    predictions=[]
    for x in range(len(testSet)):
        if k_Neg==0:
            neighbors= get_K_Neighbors(trainingSet,testSet[x],k)
        else:
            neighbors= get_negative_KNN_neighbors(trainingSet,testSet[x],k_Neg,k)
        result= getResponse(neighbors)
        predictions.append(result)
        # print('> predictions= '+repr(result) +', actual=' +repr(trainingSet[x][-1]))
    return predictions

#the followign function will be done for aech item in a trainigSet or Testingn set.
def get_negative_KNN_neighbors(trainingSet,testInstannce,k_Neg,k):
    class_names = []
    class_counter = {}
    for x in range(len(trainingSet)): # get the classes names.
        class_name = trainingSet[x][-1]
        if class_name in class_counter:
            class_counter[class_name] += 1
        else:
            class_counter[class_name] = 1
            class_names.append(class_name)
    neighbors = get_sorted_neighbors(trainingSet, testInstannce)
    #print "initial num of neighbors: "+ "%s"%len(neighbors)
    for name in class_names: #go throw aech class
        poped=0

        # for item in neighbors: # removes k_Neg nearest neighbors from the current class.
        #     if item[-1] == name:
        #         neighbors.remove(item)
        #         poped+=1
        #         #print "poped "+'%s'%name
        #         if poped == k_Neg:
        #             break
        arraySize = len(neighbors)
        i = 0
        while i < arraySize:
            if neighbors[i][-1]== name:
                del neighbors[i]
                poped+=1
                #print "poped " + '%s' % name
                if poped == k_Neg:
                    break
                arraySize -= 1
            else:
                i += 1
    #     print "------------"
    # print "%%%%%%%%%%%%%%%%%"
    #print "final num of neighbors: " + "%s" % len(neighbors)
    knn=[]
    for x in range(k): # keep only the k nearest neighbbors
        knn.append(neighbors[x])
    return knn


def noisyTrainingSet_KNN(trainingSet, training_predictions,testSet,k,withENN=0):
    print "KNN - TrainingSet with noise:\n"
    class_names = []
    trainingSet_noise_matrix = get_training_noise_matrix(trainingSet, training_predictions, class_names)
    noisyTrainingSet= getNoisyTrainingSet(trainingSet, trainingSet_noise_matrix, class_names)
    KNN(noisyTrainingSet,testSet,k,0,withENN) #run KNN with noisyTrainingSet
    if withENN:
        noisyTraining_predictions = get_predictions(noisyTrainingSet,noisyTrainingSet,k)
        ReducedNoisyTrainingSet= get_reducedTrainingSet(noisyTrainingSet,noisyTraining_predictions)
        training_accuracy = getAccuracy(ReducedNoisyTrainingSet, noisyTraining_predictions)
        test_predictions = get_predictions(ReducedNoisyTrainingSet, testSet, k)
        test_accuracy = getAccuracy(testSet, test_predictions)
        print "KNN - Reduced Noisy TrainingSet (ENN & KNN):\n"
        print ('TrainingSet Accuracy: ' + repr(training_accuracy) + '%')
        print ('TestSet Accuracy: ' + repr(test_accuracy) + '%')
        print ("---------------------------------")

def getNoisyTrainingSet(trainingSet, noiseMatrix, class_names):
    noisyTrainingSet=trainingSet[:]
    for i in range(len(trainingSet)):
        realVal = trainingSet[i][-1]
        rowIndex = class_names.index(realVal)
        #population = [1, 2, 3]
        #weights = [0.1, 0.5, 0.4]
        weights= noiseMatrix[rowIndex]
        givenval = np.random.choice(class_names, 1, p=weights)[0]
        noisyTrainingSet[i][-1]= givenval
    return noisyTrainingSet


def get_reducedTrainingSet(trainingSet,predictions):
    reducedTrainingSet= trainingSet[:]
    arraySize = len(trainingSet)
    i = 0
    while i < arraySize:
        if reducedTrainingSet[i][-1] != predictions[i]:
            del reducedTrainingSet[i]
            del predictions[i]
            arraySize -= 1
        else:
            i += 1
    return reducedTrainingSet


def KNN(trainingSet,testSet,k,withNoise=0,withENN=0,k_Neg=0):
    training_predictions= get_predictions(trainingSet, trainingSet, k,k_Neg)
    test_predictions= get_predictions(trainingSet, testSet, k,k_Neg)
    test_accuracy = getAccuracy(testSet, test_predictions)
    training_accuracy = getAccuracy(trainingSet, training_predictions)
    print ('TrainingSet Accuracy: ' + repr(training_accuracy) + '%')
    print ('TestSet Accuracy: ' + repr(test_accuracy) + '%')
    print ("---------------------------------")
    if withNoise:
        noisyTrainingSet_KNN(trainingSet, training_predictions,testSet,k,withENN)


def dataFile_main(data_file_name,k,k_Neg=0,withNoise=0,withENN=0):
    print "for K = %s, NEG_K = %s:" % (k, k_Neg)
    print ("---------------------------------")
    print "KNN - original TrainingSet:\n"
    # prepare data
    trainingSet = []
    testSet =[]
    split = 0.67
    loadDataet(data_file_name,split,trainingSet,testSet)
    #print('Train: ' + repr(len(trainingSet)))
    # print('Test: ' + repr(len(testSet)))
    KNN(trainingSet,testSet,k,withNoise,withENN) #run KNN with original data.
    if k_Neg!=0:
        print "Negative KNN:\n"
        KNN(trainingSet,testSet,k,0,0,k_Neg) #run Negative KNN with original data.


def gaussian_main(mean1,sd1,mean2,sd2,k,k_Neg=0,withNoise=0,withENN=0):
    print "for K = %s, NEG_K = %s:" % (k, k_Neg)
    print ("---------------------------------")
    print "original TrainingSet:\n"
    # Gussian number 1:
    g1_x = get_sampled_Gaussian(mean1, sd1, 1000)
    g1_y = np.ones((1000,), dtype=int)
    g1 = np.column_stack((g1_x, g1_y))
    train_g1 = g1[0:499]
    test_g1 = g1[500:999]

    # Gussian number 2:
    g2_x = get_sampled_Gaussian(mean2, sd2, 1000)
    g2_y = np.ones((1000,), dtype=int) * 2
    g2 = np.column_stack((g2_x, g2_y))
    train_g2 = g2[0:499]
    test_g2 = g2[500:999]

    trainingSet = np.vstack((train_g1, train_g2))
    testSet = np.vstack((test_g1, test_g2))
    sns.set()
    KNN(trainingSet.tolist(), testSet.tolist(),k,withNoise,withENN)
    if k_Neg!=0:
        print "NEG KNN:\n"
        KNN(trainingSet,testSet,k,0,0,k_Neg) #run Negative KNN with original data.
    train_df = pd.DataFrame(trainingSet, columns=['X', 'class'])  # Create the pandas DataFrame
    disply_bee_swarm(train_df)


#-----------------------------------------------------#

k = 3
negative_k = 2                #if eaquals zero, the Nagative KNN algorithm will not be tested.

with_noise=True
with_ENN=True

# ACTIVATE ONE FROM THE FOLLOWING OPTIONS:

#--- OPTION 1: data file ---#
dataFileName = 'iris.data'
dataFile_main(dataFileName,k,negative_k,with_noise,with_ENN)       # remove the '#' from the begining of this line to activate.

#---OPTION 2: data in a form of two Gaussians ---#
mean1= 4
sd1=2
mean2=10
sd2=2

#gaussian_main(mean1,sd1,mean2,sd2,k,negative_k,with_noise,with_ENN)         # remove the '#' from the begining of this line to activate.


