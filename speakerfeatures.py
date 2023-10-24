# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:26:59 2015

@author: Abhijeet Kumar
@code :  This program implemets feature (MFCC + delta)
         extraction process for an audio. 
@Note :  20 dim MFCC(19 mfcc coeff + 1 frame log energy)
         20 dim delta computation on MFCC features. 
@output : It returns 40 dimensional feature vectors for an audio.
"""

import numpy as np
import pywt
from sklearn import preprocessing
from python_speech_features import mfcc

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    

    #signal ,samplerate ,winlen ,winstep ,numcep 
    mfcc_feat = mfcc(audio,rate,0.025,0.01,60)
    U, S, VT = np.linalg.svd(mfcc_feat)
    num_components = 20
    svd=np.dot(U[:, :num_components], np.dot(np.diag(S[:num_components]), VT[:num_components, :]))
    return svd

def extract_features_wpt(audio,rate):
    decomposition_level = 6
    wavelet='db2'
    wp = pywt.WaveletPacket(data=audio, wavelet=wavelet, mode='symmetric', maxlevel=decomposition_level)
    
    # Create an empty matrix to store coefficients
    coeff_matrix = []

    # Iterate through nodes at the specified level
    for i, node in enumerate(wp.get_level(decomposition_level, 'freq')):
        if i==0:
            coeff_matrix = node.data[:100]
        # print(len(node.data))
        coeff_matrix=np.vstack((coeff_matrix, node.data[:100]))

    # SVD
    U, S, VT = np.linalg.svd(coeff_matrix)

    # Define the desired rank for compression
    rank = 32  # You can adjust this value

    # Reconstruct the compressed matrix
    compressed_matrix = np.dot(U[:, :rank], np.dot(np.diag(S[:rank]), VT[:rank, :]))

    return coeff_matrix


#    
if __name__ == "__main__":
     print("In main, Call extract_features(audio,signal_rate) as parameters")
     