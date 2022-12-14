#!/usr/bin/env python
# coding: utf-8

# In[1]:


import __main__ as main
IS_NOTEBOOK = not hasattr(main, '__file__')


# In[2]:


# from keras import backend as K
# cfg = K.tf.ConfigProto()
# cfg.gpu_options.allow_growth = True
# K.set_session(K.tf.Session(config=cfg))


# In[3]:


if IS_NOTEBOOK:
    get_ipython().system('nvidia-smi')
    get_ipython().system('nvidia-smi -L')


# In[4]:


import argparse

args = argparse.ArgumentParser()

args.add_argument("--loggerPrefix",type=str, default="log")

args.add_argument("--epochs",type=int,default=20)
args.add_argument("--batchSize",type=int,default=128)
args.add_argument("--trojan",type=bool,default=True)
args.add_argument("--poisonSampleCount",type=int,default=5000)

args.add_argument("--dataset",type=str,default="mnist")
# args.add_argument("--dataset",type=str,default="cifar10")

args.add_argument("--optimizer",type=str,default="sgd")
# args.add_argument("--optimizer",type=str,default="adam")

# args.add_argument("--fixedPoisonLocation",type=int,default=None)
args.add_argument("--fixedPoisonLocation",type=int,default=1)


args.add_argument("--modelSaveFile",type=str,default=None)
# args.add_argument("--modelSaveFile",type=str,default="cifarTrained-20221121-0.h5")
args.add_argument("--modelLoadFile",type=str,default=None)
# args.add_argument("--modelLoadFile",type=str,default="cifarTrained-20221121-0.h5")

args.add_argument("--modelTrain",type=bool,default=True)


args.add_argument("--experimentType",type=str,default="shuffled")
# args.add_argument("--experimentType",type=str,default="fullBatch")
# args.add_argument("--experimentType",type=str,default="percentageOfBatch")



args.add_argument("--poisonColorR",type=int, default=128)
args.add_argument("--poisonColorG",type=int, default=128)
args.add_argument("--poisonColorB",type=int, default=128)


args.add_argument("--diversityFactor",type=str, default=None, help="Give a dictionary using : and , within double quotes.")
# args.add_argument("--diversityFactor",type=str, default="type:multipleLocations,noLocations:2")
# args.add_argument("--diversityFactor",type=str, default="type:multipleLocations,noLocations:3")
# args.add_argument("--diversityFactor",type=str, default="type:locationVariance,locationVariance:2")
# args.add_argument("--diversityFactor",type=str, default="type:locationVariance,locationVariance:5")
# args.add_argument("--diversityFactor",type=str, default="type:locationVariance,locationVariance:8")





# In[5]:


if IS_NOTEBOOK: args = args.parse_args(args=[])
else: args = args.parse_args()

    
LOGGER_PREFIX = args.loggerPrefix
logDict = {}
logDict["mergedTrainLoss"]=[]
logDict["cleanTestLoss"]=[]
logDict["mergedTrainAcc"]=[]
logDict["cleanTestAcc"]=[]
logDict["trojanEffectivenessLoss"]=[]
logDict["trojanEffectivenessAcc"]=[]


# In[6]:


EPOCHS = args.epochs
BATCH_SIZE = args.batchSize
TROJAN = args.trojan
DATASET = args.dataset
POISON_SAMPLE_COUNT = args.poisonSampleCount
OPTIMIZER = args.optimizer


FIXED_POISON_LOCATION = args.fixedPoisonLocation


MODEL_SAVE = not (args.modelSaveFile==None)
MODEL_LOAD = not (args.modelLoadFile==None)
MODEL_TRAIN = args.modelTrain

if MODEL_SAVE: MODEL_FILE_NAME = args.modelSaveFile
elif MODEL_LOAD: MODEL_FILE_NAME = args.modelLoadFile

COUNTER_imagesSaved = 0
EXPERIMENT_TYPE = args.experimentType


POISON_COLOR_R = args.poisonColorR
POISON_COLOR_G = args.poisonColorG
POISON_COLOR_B = args.poisonColorB

DIVERSITY_FACTOR_DICT = {}
if args.diversityFactor==None:
    DIVERSITY_FACTOR_DICT=None
else:
    keyVal = args.diversityFactor.strip().split(",")
    for kv in keyVal:
        k,v=kv.split(":")
        try:
            DIVERSITY_FACTOR_DICT[k]=int(v)
        except:
            DIVERSITY_FACTOR_DICT[k]=v
        
    print("Diversity Factor Dict=")
    print(DIVERSITY_FACTOR_DICT)


# In[7]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Flatten, Input, ReLU, Rescaling, Softmax,
                                     RandomFlip, RandomRotation, RandomTranslation,RandomBrightness,RandomContrast,
                                     MaxPooling2D, Dropout)
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam, SGD


# tf.keras.backend.set_image_data_format("channels_first")







# In[8]:


print(tf.keras.backend.image_data_format())
print(tf.config.list_physical_devices('GPU'))

gpu_devices = tf.config.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

print(tf.config.list_physical_devices())


# In[9]:


# %pip install numba
from numba import cuda
def clearGPU(modelsInGPU=None,working=False):
    if not working: return modelsInGPU
    # tf.keras.backend.clear_session()
    # del model
    
    if not modelsInGPU==None:
        if type(modelsInGPU) == list:
            _=0
        else:
            modelsInGPU.save("tempSaveAndLoad.h5")
            modelsToReturn = tf.keras.models.load_model("tempSaveAndLoad.h5")
    
    
    cuda.select_device(0)
    cuda.close()
    return modelsToReturn


# In[10]:


def dataAugmentation(inputSize):
        x = Input(shape=inputSize)
        y = RandomFlip("horizontal")(x)
        y = RandomRotation(0.2)(y)
        # y = RandomZoom(0.2)(y)
        # y = RandomCrop(inputSize[1], inputSize[2])(y)
        # y = RandomContrast(0.2)(y)
        # y = RandomTranslation(0.2, 0.2)(y)
        # y = RandomBrightness(0.2)(y)
        model = tf.keras.Model(inputs=x, outputs=y)
        return model





def printFrequenciesOfOneHotGroundTruth(y):
        y = np.argmax(y,axis=1)
        unique, counts = np.unique(y, return_counts=True)
        print(dict(zip(unique, counts)))



def saveNumpyAsImage(x,fileName):
        x = np.squeeze(x)
        x = x#*255
        x = x.astype(np.uint8)
        img = Image.fromarray(x, 'RGB')
        img.save(fileName)


# In[11]:


def smallCNN(inputSize):
        x = Input(shape=inputSize)
        # y0 = Rescaling(1./255)(x)
        y0 = x
        y1 = Conv2D(16, 3, padding='same')(y0)
        y2 = BatchNormalization()(y1)
        y3 = ReLU()(y2)
        y4 = Conv2D(32, 4, padding='same', strides=2)(y3)
        y5 = BatchNormalization()(y4)
        y6 = ReLU()(y5)
        y7 = Conv2D(32, 4, padding='same', strides=2)(y6)
        y8 = BatchNormalization()(y7)
        y9 = ReLU()(y8)
        y10 = Flatten()(y9)
        y11 = Dense(128)(y10)
        y12 = BatchNormalization()(y11)
        y13 = ReLU()(y12)
        y14 = Dense(10)(y13)
        y15 = Softmax()(y14)
        y = y15
        model = tf.keras.Model(inputs=x, outputs=y)
        return model


# In[12]:


def smallCNN2(inputSize):
        # 100 Epoch accuracy = 83.450
        # As per https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
        model = Sequential()
        model.add(Input(shape=inputSize))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
        return model




# In[13]:


def mnistCNN(inputSize):
    #https://www.kaggle.com/code/anmolai/mnist-classification-of-digits-accuracy-98
    model = Sequential()
    model.add(Input(shape=inputSize))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
    return model


# In[14]:


def shuffle2(x,y,lists=False):
    if not lists:
        assert x.shape[0]==y.shape[0], "Shuffling different sized arrays together."
        randomPermutation = np.random.shuffle(np.arange(x.shape[0]))
        x = x[randomPermutation]
        y = y[randomPermutation]
        
        x,y = np.squeeze(x,axis=0), np.squeeze(y,axis=0)
    
    else:
        assert len(x)==len(y),"Shuffling different sized lists together."
        randomPermutation = np.random.shuffle(np.arange(len(x)))
        
        xNew = [x[randomPermutation[i]] for i in range(randomPermutation.shape[0])]
        yNew = [y[randomPermutation[i]] for i in range(randomPermutation.shape[0])]
        
        x,y = xNew,yNew
    return x,y
    


# In[15]:


# def putShape(inputImages,locations, poisonType="triangle"):
    
    


# In[16]:


def showNImagesWithLabels(startIdx,N,X,Y):
#     import matplotlib.pyplot as plt
#     plt.rcParams['figure.figsize'] = [10, 5]

    ansY = str(np.argmax(Y[startIdx]))
    ansX = X[startIdx]
    for t in range(1,N):
        ansY = ansY + " " + str(np.argmax(Y[startIdx + t]))
        ansX = np.concatenate((ansX,X[startIdx + t]), axis=1)
    
    print("LABELS : ",ansY)
    print("Min: ",np.min(ansX),"Max: ",np.max(ansX), "Size of all images:",ansX.shape)
    showNumpyAsImage(ansX)
    


# In[17]:


def poisonUtil(inputImages, xLocations, yLocations, offsetsXY, colors, diffColors=False):
    print(inputImages.dtype)
    N = inputImages.shape[0]
    H = inputImages.shape[1]
    W = inputImages.shape[2]
    
    
    for oxy in offsetsXY:
        if not diffColors:
            inputImages[np.arange(N), xLocations + oxy[0], yLocations + oxy[1], 0] = colors[0]
            inputImages[np.arange(N), xLocations + oxy[0], yLocations + oxy[1], 1] = colors[1]
            inputImages[np.arange(N), xLocations + oxy[0], yLocations + oxy[1], 2] = colors[2]
        if diffColors:
            inputImages[np.arange(N), xLocations + oxy[0], yLocations + oxy[1], 0] = colors[:,0]
            inputImages[np.arange(N), xLocations + oxy[0], yLocations + oxy[1], 1] = colors[:,1]
            inputImages[np.arange(N), xLocations + oxy[0], yLocations + oxy[1], 2] = colors[:,2]
        
        
    return inputImages


# In[18]:


def poisonDatasetMerger(inputImages,listOfDictsOfIdxXY,offsetsXY):
    N = inputImages.shape[0]
    xLocations = np.zeros((N),dtype=int)
    yLocations = np.zeros((N),dtype=int)
    colors = np.zeros((N,3),dtype=int)
    for di in listOfDictsOfIdxXY:
        xLocations[di["idx"]]=di["X"]
        yLocations[di["idx"]]=di["Y"]
        colors[di["idx"]]=di["colors"]
    return poisonUtil(inputImages,xLocations,yLocations,offsetsXY,colors,diffColors=True)
    
    


# In[19]:


def poisonDataset(inputImages,poisonLabel=0,poisonType="traingle",fixedLocation=None, redPixel=False, diversityFactor=None):        
        inputImages = np.array(inputImages)
        
        print(inputImages.dtype)
        N = inputImages.shape[0]
        H = inputImages.shape[1]
        W = inputImages.shape[2]

        shapes={}
        shapes["traingle"] = [[0,0],[1,0],[0,1]]
        shapes["square"] = [[0,0],[1,0],[0,1],[1,1]]
        shapes["dialatedSquare"] = [[0,0],[2,0],[0,2],[2,2]]
        
        
        if diversityFactor==None:


            
            xIdx = np.full((N), fixedLocation[0], dtype=int)
            yIdx = np.full((N), fixedLocation[1], dtype=int)
            idx = np.arange(N, dtype=int)
            
            col = np.tile(np.array([POISON_COLOR_R,POISON_COLOR_G,POISON_COLOR_B],dtype=int),N).reshape((N,3))
            inputImages=poisonDatasetMerger(inputImages,[{"idx":idx,"X":xIdx,"Y":yIdx,"colors":col}],shapes[poisonType])
            
#             inputImages = poisonUtil(inputImages, xIdx, yIdx,shapes[poisonType] ,\
#                                      [POISON_COLOR_R,POISON_COLOR_G,POISON_COLOR_B])

        else:
            if diversityFactor["type"] == "multipleLocations":
                
                if diversityFactor["noLocations"]==2:
                    idx = np.random.shuffle(np.arange(N,dtype=int))
                    idxA = idx[:idx.shape[0]//2]
                    idxB = idx[idx.shape[0]//2:]

                    xIdxA = np.full((N//2), 2, dtype=int)
                    xIdxB = np.full((N//2), 2, dtype=int)

                    yIdxA = np.full((N//2), 16, dtype=int)
                    yIdxB = np.full((N//2), 16, dtype=int)

                    col = np.tile(np.array([POISON_COLOR_R,POISON_COLOR_G,POISON_COLOR_B],dtype=int),N).reshape((N,3))
                    colA = col[:col.shape[0]//2]
                    colB = col[col.shape[0]//2:]


                    listOfDicts = [{"idx":idxA,"X":xIdxA,"Y":YIdxA,"colors":colA},\
                                  {"idx":idxB,"X":xIdxB,"Y":YIdxB,"colors":colB}]

                    inputImages = poisonDatasetMerger(inputImages,listOfDicts,shapes[poisonType])
                    
                
                elif diversityFactor["noLocations"] == 3:
                    assert False, "Implement this!"

            elif diversityFactor["type"] == "locationVariance":
                v = diversityFactor["locationVariance"]
                
                x = fixedLocation[0]
                y = fixedLocation[1]
                
                xMin = max(2,x-v)
                xMax = min(inputImages.shape[1]-3,x+v)
                
                
                yMin = max(2,y-v)
                yMax = min(inputImages.shape[2]-3,y+v)
                
                
                idx = np.arange((N),dtype=int)
                xIdx = np.random.random_integers(xMin,high=xMax,size=(N))
                yIdx = np.random.random_integers(yMin,high=yMax,size=(N))
                
                
                col = np.tile(np.array([POISON_COLOR_R,POISON_COLOR_G,POISON_COLOR_B],dtype=int),N).reshape((N,3))
                inputImages=poisonDatasetMerger(inputImages,[{"idx":idx,"X":xIdx,"Y":yIdx,"colors":col}],shapes[poisonType])
                
                
                
                
            
            
        return inputImages, tf.keras.utils.to_categorical(poisonLabel*np.ones(N), num_classes=10,dtype='float32')






# In[20]:


def appendPoisonToDataset(x,y,poisonLabel=0,poisonType="traingle",poisonSampleCount=1000,fixedLocation=None,\
                         experimentType = None, batchSize = 32, verbose=False,diversityFactor=None):
        
        assert experimentType in ["shuffled", "fullBatch", "percentageOfBatch"], "Wrong experiment type"
        
        
        print("DEBUG: x.shape, y.shape",x.shape, y.shape)
        
        if verbose:
            print("Show before shuffling")
            showNImagesWithLabels(0,10,x,y)
#             for t in range(10):
#                 print(t, np.argmax(y[t]))
#                 showNumpyAsImage(x[t])
        
        x,y = shuffle2(x,y)
       
    
        if verbose:
            print("Show after shuffling")    
            showNImagesWithLabels(0,10,x,y)
#             for t in range(10):
#                 print(t, np.argmax(y[t]))
#                 showNumpyAsImage(x[t])
        
        xPoison, yPoison = poisonDataset(x[:poisonSampleCount],poisonLabel=poisonLabel,\
                                         poisonType=poisonType,fixedLocation=fixedLocation,diversityFactor=diversityFactor)
        
        print("DEBUG: x.shape, y.shape, xPoison.shape, yPoison.shape",x.shape, y.shape, xPoison.shape, yPoison.shape)
        
        if experimentType=="shuffled":
            xNew = np.concatenate((x,xPoison),axis=0)
            yNew = np.concatenate((y,yPoison),axis=0)
        
        elif experimentType=="fullBatch":
            idxStart = 0
            xBatches = []
            yBatches = []
            while idxStart < poisonSampleCount:
                idxEnd = idxStart + batchSize//2
#                 print("DEBUG: idxEnd=idxStart + batchSize/2 = ",idxEnd)
                thisBatchX = np.concatenate((x[idxStart:idxEnd],xPoison[idxStart:idxEnd]),axis=0)
                thisBatchY = np.concatenate((y[idxStart:idxEnd],yPoison[idxStart:idxEnd]),axis=0)
                
                xBatches.append(thisBatchX)
                yBatches.append(thisBatchY)
                
                
                idxStart += batchSize//2
            
            if verbose:
                print("Before adding all the clean labels")
                print("No of Batches",len(xBatches),len(yBatches))
                print(" ".join([str(b.shape) for b in xBatches]))
            
            while idxStart< x.shape[0]:
                idxEnd = idxStart + batchSize
                xBatches.append(x[idxStart:idxEnd])
                yBatches.append(y[idxStart:idxEnd])
                idxStart = idxEnd
            
            if verbose:
                print("After adding all the clean labels")
                print("No of Batches",len(xBatches),len(yBatches))
                print(" ".join([str(b.shape) for b in xBatches]))
            
            
            xNew = np.concatenate(xBatches,axis=0)
            yNew = np.concatenate(yBatches,axis=0)
            
            
        elif experimentType=="percentageOfBatch":
            idxCleanStart = 0
            idxPoisonStart = 0
            
            
            
            noBatches = int((x.shape[0] + xPoison.shape[0])/batchSize)
            poisonSamplesPerBatch = int(xPoison.shape[0]/noBatches)
            print("DEBUG:  poisonSamplesPerBatch=",poisonSamplesPerBatch)
            
            assert poisonSamplesPerBatch>0, "Not even one poison per batch"
            
            xBatches = []
            yBatches = []
            
            while idxCleanStart < x.shape[0] and idxPoisonStart < xPoison.shape[0]:
                idxPoisonEnd  =  idxPoisonStart + poisonSamplesPerBatch
                idxCleanEnd = idxCleanStart + poisonSamplesPerBatch 
                
                thisBatchX = np.concatenate((x[idxCleanStart:idxCleanEnd],xPoison[idxPoisonStart:idxPoisonEnd]),axis=0)
                thisBatchY = np.concatenate((y[idxCleanStart:idxCleanEnd],yPoison[idxPoisonStart:idxPoisonEnd]),axis=0)
                
                xBatches.append(thisBatchX)
                yBatches.append(thisBatchY)
                
                idxPoisonStart = idxPoisonEnd
                idxCleanStart = idxCleanEnd
            
            batchIdx = 0
            while idxCleanStart < x.shape[0]:
                idxCleanEnd = idxCleanStart + batchSize - xBatches[batchIdx].shape[0]
                thisBatchX = np.concatenate((xBatches[batchIdx],x[idxCleanStart:idxCleanEnd]),axis=0)
                thisBatchY = np.concatenate((yBatches[batchIdx],y[idxCleanStart:idxCleanEnd]),axis=0)
                
                xBatches[batchIdx] = thisBatchX
                yBatches[batchIdx] = thisBatchY
                
                batchIdx+=1
                idxCleanStart=idxCleanEnd
            
        
            xNew = np.concatenate(xBatches,axis=0)
            yNew = np.concatenate(yBatches,axis=0)
        else:
            assert False, "ERROR"

            
        if verbose:
            print("Show concatenated")    
            showNImagesWithLabels(x.shape[0]-5,10,xNew,yNew)
#             for t in range(10):
#                 print(t, np.argmax(yNew[x.shape[0]-5+t]))
#                 showNumpyAsImage(xNew[x.shape[0]-5+t])
            
            
        toReturn = {"mergedX":xNew,"mergedY":yNew,"poisonX":xPoison,"poisonY":yPoison,"cleanX":x,"cleanY":y}
        return toReturn


# In[21]:


import matplotlib.pyplot as plt
def showNumpyAsImage(x):
        x = np.squeeze(x)
        x = x*255
        x = x.astype(np.uint8)
        if IS_NOTEBOOK:
#             plt.figure(figsize=(1, 1))
            plt.imshow(x)
            plt.show()
        else:
            _=None


# In[22]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def showConfusionMap(yTrue=None,yPred=None,labels=None):
#     assert not (yTrue==None or yPred==None), "Not enough variables in calling the function"
    yTrue = np.argmax(yTrue,axis=-1)
    yPred = np.argmax(yPred,axis=-1)
    
    print(yTrue.shape, yTrue[:10])
    print(yPred.shape, yPred[:10])
    cm = confusion_matrix(yTrue, yPred,labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    if IS_NOTEBOOK:
        plt.show()
    
    
        


# In[23]:


if DATASET=="cifar10":
    (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    INPUT_SIZE = (32,32,3)
elif DATASET=="mnist":
    (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    INPUT_SIZE = (28,28,1)
    
#     print(xTrain[0])
    
    print(xTrain.shape)
    showNumpyAsImage(xTrain[0]/255.0)
    xTrain=np.stack((xTrain,xTrain,xTrain),axis=3)
    print(xTrain.shape)
    
    
    xTest=np.stack((xTest,xTest,xTest),axis=3)
    
    showNumpyAsImage(xTrain[0]/255.0)

    
yTrain = tf.keras.utils.to_categorical(yTrain,num_classes=10, dtype='float32')
yTest = tf.keras.utils.to_categorical(yTest,num_classes=10, dtype='float32')


# In[24]:


def printShapesDictOfAr(dictOfAr):
    toPrint = ""
    for k in dictOfAr.keys():
        toPrint += str(k) + " " + str(dictOfAr[k].shape) + " | "
    print(toPrint)


# In[25]:


# Testing the trojan dataset creation fucntions
# "shuffled", "fullBatch", "percentageOfBatch"

# mergedPoisonCleanData = appendPoisonToDataset(xTrain,yTrain,\
#         poisonLabel=0,poisonType="traingle",\
#         poisonSampleCount=POISON_SAMPLE_COUNT, fixedLocation=[FIXED_POISON_LOCATION,FIXED_POISON_LOCATION],\
#         experimentType = "shuffled",verbose=False)

# printShapesDictOfAr(mergedPoisonCleanData)

# print("Test end")


# In[26]:


if DATASET=="mnist":
    model = mnistCNN((28,28,3))
elif DATASET=="cifar10":
    model = smallCNN2(INPUT_SIZE)
else:
    assert False, "Problem!"
    
model.summary()


# In[27]:


if False:
    augmentationModel = dataAugmentation(INPUT_SIZE)
    augmentationModel.summary()


    modelToTrain = tf.keras.Sequential([augmentationModel, model])
    modelToTrain.summary()    


# In[28]:


if OPTIMIZER=="sgd":
    opt = SGD(learning_rate=0.001, momentum=0.9)
elif OPTIMIZER=="adam":
    opt= Adam(learning_rate=0.001)
else:
    assert False, "Wrong optimizer"
    
model.compile(optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])


# In[29]:


# if TROJAN:
print("Trojan (poison) dataset is being created")

mergedPoisonCleanData = appendPoisonToDataset(xTrain,yTrain,\
        poisonLabel=0,poisonType="traingle",\
        poisonSampleCount=POISON_SAMPLE_COUNT, fixedLocation=[FIXED_POISON_LOCATION,FIXED_POISON_LOCATION],\
        experimentType = "shuffled", diversityFactor = DIVERSITY_FACTOR_DICT)
xTrain = mergedPoisonCleanData["mergedX"]
yTrain = mergedPoisonCleanData["mergedY"]


print("Train shapes", xTrain.shape, yTrain.shape)
print("Test shapes", xTest.shape, yTest.shape)


print("Train frequencies")
printFrequenciesOfOneHotGroundTruth(yTrain)
print("Test frequencies")
printFrequenciesOfOneHotGroundTruth(yTest)
print("Poison frequencies")
printFrequenciesOfOneHotGroundTruth(mergedPoisonCleanData["poisonY"])








# In[30]:


class MultipleValidationSetsCallback(tf.keras.callbacks.Callback):
    def __init__(self,model,xyPairs,xyPairNames):
        self.model = model
        self.xyPairs = xyPairs
        self.xyPairNames = xyPairNames
    
    def on_epoch_end(self, epoch, logs=None):
        ans = ""
        for xyIdx in range(len(self.xyPairs)):
            xy = self.xyPairs[xyIdx]
            evalRes = self.model.evaluate(xy[0],xy[1],return_dict=True,verbose=0)
            for k in evalRes.keys():
                evalRes[k] = int(evalRes[k]*1000)/1000
                
                if k=="loss":
                    logDict[self.xyPairNames[xyIdx]+"Loss"].append(evalRes[k])
                elif k=="accuracy":
                    logDict[self.xyPairNames[xyIdx]+"Acc"].append(evalRes[k])
            
            ans = ans + " " + str(evalRes)
        print("Eval acc: ",ans)
        


# In[ ]:





# In[31]:


printFrequenciesOfOneHotGroundTruth(mergedPoisonCleanData["mergedY"])
showNImagesWithLabels(mergedPoisonCleanData["cleanX"].shape[0] +50 ,10,\
                      mergedPoisonCleanData["mergedX"],mergedPoisonCleanData["mergedY"])

printFrequenciesOfOneHotGroundTruth(mergedPoisonCleanData["poisonY"])
showNImagesWithLabels(0 ,10,\
                      mergedPoisonCleanData["poisonX"],mergedPoisonCleanData["poisonY"])

printFrequenciesOfOneHotGroundTruth(mergedPoisonCleanData["cleanY"])
showNImagesWithLabels(0 ,10,\
                      mergedPoisonCleanData["cleanX"],mergedPoisonCleanData["cleanY"])

# print(np.argmax(mergedPoisonCleanData["mergedY"],axis=-1))


# In[32]:


# model.fit(xTrain/255.0, yTrain)

if MODEL_LOAD:
    model=tf.keras.models.load_model(MODEL_FILE_NAME)
    print("Loaded model ",MODEL_FILE_NAME)

if MODEL_TRAIN:
    callBack = MultipleValidationSetsCallback(model,\
        [[xTest/255.0,yTest],\
        [mergedPoisonCleanData["poisonX"]/255.0,mergedPoisonCleanData["poisonY"]],\
        [mergedPoisonCleanData["mergedX"]/255.0,mergedPoisonCleanData["mergedY"]]],
        ["cleanTest","trojanEffectiveness","mergedTrain"])
    
    
    
    if EXPERIMENT_TYPE=="shuffled":
        print("XXXXXX")
#         model.fit(xTrain/255.0, yTrain, epochs=EPOCHS, batch_size=BATCH_SIZE,  callbacks=[callBack],shuffle=True)
        model.fit(mergedPoisonCleanData["mergedX"]/255.0, mergedPoisonCleanData["mergedY"],\
                  epochs=EPOCHS, batch_size=BATCH_SIZE,  callbacks=[callBack],shuffle=True, verbose=1)
    else:
        model.fit(mergedPoisonCleanData["mergedX"]/255.0, mergedPoisonCleanData["mergedY"], epochs=EPOCHS,\
                  batch_size=BATCH_SIZE, callbacks=[callBack],shuffle=False, verbose=1)
if MODEL_SAVE:
    model.save(MODEL_FILE_NAME)
    print("Saved model ",MODEL_FILE_NAME)


# In[33]:


print("Clean test accuracy")
model.evaluate(xTest/255.0, yTest, batch_size=BATCH_SIZE)
print("Poison test accuracy")
model.evaluate(mergedPoisonCleanData["poisonX"]/255.0, mergedPoisonCleanData["poisonY"], batch_size=BATCH_SIZE)

print("End of the program")


# In[34]:


print(xTest.shape)
print(yTest.shape)
print(mergedPoisonCleanData["poisonX"].shape)
print(mergedPoisonCleanData["poisonY"].shape)


# In[35]:


IDX = 109

mergedIDX = IDX + mergedPoisonCleanData["cleanY"].shape[0]


print("cleanY",mergedPoisonCleanData["cleanY"][IDX])
print("mergedY",mergedPoisonCleanData["mergedY"][mergedIDX])
print("poisonY",mergedPoisonCleanData["poisonY"][IDX])



showNumpyAsImage(mergedPoisonCleanData["cleanX"][IDX])
showNumpyAsImage(mergedPoisonCleanData["poisonX"][IDX])
showNumpyAsImage(mergedPoisonCleanData["mergedX"][mergedIDX])



# In[36]:


showConfusionMap(yTrue=mergedPoisonCleanData["cleanY"],yPred=model.predict(mergedPoisonCleanData["cleanX"]/255.0),labels=np.arange(10))
showConfusionMap(yTrue=mergedPoisonCleanData["mergedY"],yPred=model.predict(mergedPoisonCleanData["mergedX"]/255.0),labels=np.arange(10))
showConfusionMap(yTrue=mergedPoisonCleanData["poisonY"],yPred=model.predict(mergedPoisonCleanData["poisonX"]/255.0),labels=np.arange(10))


# In[37]:


print("IMPORTANT RESULTS TO SAVE")
LogDictSaveFileLocation = "{}-allLogs.npy".format(LOGGER_PREFIX)
np.save(LogDictSaveFileLocation,logDict)
print("Dict saved to : {}".format(LogDictSaveFileLocation))


# In[38]:


print("END OF PROGRAM")




# In[ ]:




