#train_models.py

import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")


#path to training data
source   = "development_set\\"   

#path where training speakers will be saved
dest = "speaker_models\\"

train_file = "development_set_enroll.txt"        


file_paths = open(train_file,'r')

count = 1

# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print(path)
    
    # read the audio
    sr,audio = read(source + path)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    if count == 5:    
        #gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
        #gmm.fit(features)
        model = mixture.GaussianMixture(n_components = 16, covariance_type='diag',n_init = 3)
        model.fit(features)
        
        # dumping the trained gaussian model
        os.makedirs(dest, exist_ok=True)
        # Create the pickled file path
        picklefile = path.split("-")[0] + ".gmm"

        # Serialize and save the 'model' object to a file using binary write mode ('wb')
        with open(dest + picklefile, 'wb') as file:
            pickle.dump(model, file)
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
    