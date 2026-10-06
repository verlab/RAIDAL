import os
import torch

import numpy as np
import torch.nn.functional as F

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def normalizeEmbeddings(feature_dict, device):
    """L2-normalizes the per-frame feature vectors of every video in a set.
 
    Each video's embeddings are moved to the GPU and L2-normalized along the
    feature dimension, so every frame vector has unit norm and Euclidean distance
    between frames behaves like a cosine-distance comparison. Results are then
    moved back to the CPU as numpy arrays.
 
    Args:
        feature_dict (dict): Maps video id -> array/tensor of shape (T, D), the
            per-frame feature vectors for that video (T frames, D-dim embeddings).
        device (str): The torch device used to run the normalization.
 
    Returns:
        dict: Maps video id -> numpy array of shape (T, D) with L2-normalized,
            embeddings. Keeping the result on the CPU matches what
            the coreset() function expects: it moves each video's embeddings back to the GPU
            one at a time, instead of requiring every video to be resident on the
            GPU simultaneously.
    """


    normDict = {}
    for vid, embeddings in feature_dict.items():
        # Pass to GPU and normalize per-frame
        embeddings = torch.as_tensor(embeddings).float().to(device)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Send back to CPU to avoid breaking the coreset function.
        normDict[vid] = embeddings.cpu().numpy() 
    
    return normDict


def readFeaturesFromFile(featuresPath: str, featureList)-> tuple:
    """
    This function reads the .npy files in featuresPath and returns them as a dict, 
    where the keys are the names of the videos, and the values are the extracted features.

    Args:
        featuresPath   (str): The path to the extracted features
    """

    # The keys are the names of the videos, and the values are the extracted features
    features = dict()
    
    # Get the file paths for the features in the LABELED set
    for file in os.listdir(featuresPath):
        filePath    = os.path.join(featuresPath, file)

        # Strip the string and the _features.npy suffix to get only the name of the video.
        # The processed string will be something like "01April_2010_Thursday_heute_default-3"
        fileName = filePath.split("/")[-1].removesuffix("_features.npy")

        features[fileName] = dict()

        fileContent = np.load(filePath, allow_pickle=True)
        
        # Get the features from the .npy file
        for featureName in featureList:
                features[fileName][featureName] = fileContent.item()[featureName]


    return features
