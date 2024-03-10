import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from src import *

from src.ppn2v.unet import UNet

from src.ppn2v.pn2v import utils
from src.ppn2v.pn2v import training
from tifffile import imread
# See if we can use a GPU
device=utils.getDevice()


# Download data
import urllib
import zipfile

# if not os.path.isdir('../../../data'):
#     os.mkdir('../../../data')

# zipPath="../../../data/Convallaria_diaphragm.zip"
# if not os.path.exists(zipPath):  
#     data = urllib.request.urlretrieve('https://zenodo.org/record/5156913/files/Convallaria_diaphragm.zip?download=1', zipPath)
#     with zipfile.ZipFile(zipPath, 'r') as zip_ref:
#         zip_ref.extractall("../../../data")


path=os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'big_data_small', 'good_sample-unidentified')
fileName='good_sample-unidentified.tif'
dataName='good_sample-unidentified' # This will be used to name the noise2void model


data=imread(path+fileName)
nameModel=dataName+'_n2v'


# The N2V network requires only a single output unit per pixel
net = UNet(1, depth=3)

# Split training and validation data.
my_train_data=data[:-5].copy()
my_val_data=data[-5:].copy()

# Start training.
trainHist, valHist = training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                                           postfix= nameModel, directory=path, noiseModel=None,
                                           device=device, numOfEpochs= 200, stepsPerEpoch = 10, 
                                           virtualBatchSize=20, batchSize=1, learningRate=1e-3)


# Let's look at the training and validation loss
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(valHist, label='validation loss')
plt.plot(trainHist, label='training loss')
plt.legend()
plt.show()