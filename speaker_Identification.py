#speaker_Identification.py

import os
import pickle
import numpy as np
from scipy.io import wavfile
from sklearn import mixture
from speakerfeatures import extract_features
from speakerfeatures import extract_features_wpt
from sklearn.metrics import accuracy_score

# step 1
# training

#path to training data
source   = "development_set\\"   
train_file = "development_set_enroll.txt"        
file_paths = open(train_file,'r')
count = 1
models = []
speakers=[]
# Extracting features for each speaker (5 files per speakers)
for path in file_paths:
    path = path.strip()
    
    # read the audio
    samplerate,audio = wavfile.read(source + path)
    
    # extract wpt features
    features   = extract_features_wpt(audio,samplerate)
    
    # do model training, only model one file which is the 5th file
    if count == 5:
        model = mixture.GaussianMixture(n_components = 1)
        model.fit(features)
        models.append(model)
        speakers.append(path.split("-")[0])
        count = 0
    count = count + 1


# step 2
# testing

modelpath = "speaker_models\\"
test_file = "development_set_test.txt"        
file_paths = open(test_file,'r')

true_labels = []
predicted_labels = []
# Read the test directory and get the list of test audio files 
for path in file_paths:   
    
    path = path.strip()
    # Extract the name as flag, and take as the true lable set 
    true_labels.append(path.split('-')[0])
    # read the audio
    samplerate,audio = wavfile.read(source + path)
    # extract wpt features
    features   = extract_features_wpt(audio,samplerate)
    
    log_likelihood = np.zeros(len(models)) 

    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score(features))
        log_likelihood[i] = scores.sum()
    
    # find the most similar one
    winner = np.argmax(log_likelihood)

    # put in the predict set
    predicted_labels.append(speakers[winner])

# calculate the accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)