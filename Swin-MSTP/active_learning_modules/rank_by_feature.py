import numpy as np
import json
import os

from collections import defaultdict

from active_learning_modules.raidal import raidal

from active_learning_modules.uncast import uncast
from active_learning_modules.entropy import entropy
from active_learning_modules.global_pooled_coreset import globalPooledCoreset
from active_learning_modules.soft_attention_coreset import softAttentionCoreset

from active_learning_modules.utils import readFeaturesFromFile


def getGloss2IndexDict(datasetBase):
    g2i = np.load(f"./preprocess/{datasetBase}/gloss_dict.npy", allow_pickle=True).item()
    g2i = { gloss: index[0] for gloss, index in g2i.items() }

    return g2i
    

def rankSimiliratyByFeatures(labeledFeaturesPath: str, unlabeledFeaturesPath: str, strategy: str, saveFolder: str, device: str, 
                            framesInVideo: dict, datasetBase:str, frameBudget=None) -> dict:
    """
    Given a set of features for the labeled and unlabeled sets, this function ranks the videos and returns a videoRank
    dictionary containing the videos and their scores sorted by most informative to least informative.

    Args:
        labeledSet   (dict): The dict containing the features for the labeled set.   Key = video name, Value = features
        featuresUnlabeledSet (dict): The dict containing the features for the unlabeled set. Key = video name, Value = features, confidence
        strategy              (str): The selection strategy.
        saveFolder            (str): The folder where to save the rank for best samples

    Returns:
        dict: The ranking for every sample in the unlabeled set.
    """
    # For now, each sample has a default ranking score of "0"
    videoRank       = defaultdict(lambda: 0)

    if str(device).isnumeric():
        device = f"cuda:{device}"
    
    gloss2IndexDict  = getGloss2IndexDict(datasetBase)

    if strategy == 'raidal':
        labeledSet     = readFeaturesFromFile(labeledFeaturesPath,   ["biLSTMFeaturesNoDropout", "ctcTimestepsNoDropout"])
        unlabeledSet   = readFeaturesFromFile(unlabeledFeaturesPath, ["biLSTMFeaturesNoDropout", "ctcTimestepsNoDropout"])
        videoRank = raidal(     unlabeledSet,
                                labeledSet,
                                device,
                                framesInVideo,
                                temporalExpansionRadius=1,
                                numHypothesesUsed=10,
                                frameBudget=frameBudget
                                )

    elif strategy == 'global-pooled-coreset':
        labeledSet      = readFeaturesFromFile(labeledFeaturesPath,   ["biLSTMFeaturesNoDropout"])
        unlabeledSet    = readFeaturesFromFile(unlabeledFeaturesPath, ["biLSTMFeaturesNoDropout"])
        videoRank       = globalPooledCoreset(
                                                unlabeledSet,
                                                labeledSet,
                                                device,
                                                framesInVideo,
                                                frameBudget
                                            )

    elif strategy == 'soft-attention-coreset':
        labeledSet     = readFeaturesFromFile(labeledFeaturesPath,   ["biLSTMFeaturesNoDropout", "ctcTimestepsNoDropout", "classifierFeaturesNoDropout"])
        unlabeledSet   = readFeaturesFromFile(unlabeledFeaturesPath, ["biLSTMFeaturesNoDropout", "ctcTimestepsNoDropout", "classifierFeaturesNoDropout"])
        videoRank      = softAttentionCoreset(
                                                unlabeledSet,
                                                labeledSet,
                                                device,
                                                framesInVideo,
                                                frameBudget
                                            )

    elif strategy == 'entropy':
        unlabeledSet    = readFeaturesFromFile(unlabeledFeaturesPath, ["classifierFeaturesNoDropout"])
        videoRank       = entropy(unlabeledSet)

    elif strategy == 'uncast':
        unlabeledSet    = readFeaturesFromFile(unlabeledFeaturesPath, ["classifierFeaturesNoDropout", "ctcDecodedSeqsNoDropout", "classifierFeaturesWithDropout", "ctcDecodedSeqsWithDropout"])
        videoRank       = uncast(
                                    unlabeledSet,
                                    gloss2IndexDict,
                                    device
                                )

    else:
        raise NotImplementedError(f"strategy = {strategy} not implemented!")

    # Order by most informative to least informative
    videoRank = dict(sorted(videoRank.items(), key=lambda x: x[1], reverse=True))

    # Convert from numpy float to regular float. This is because json.dump() can't save numpy floats :(
    videoRank = { k: float(v) for k, v in videoRank.items() }


    # Save the similarity ranking
    savePath = os.path.join(saveFolder, "videoRank.json")
    with open(savePath, "w") as filePointer:
        json.dump(videoRank, filePointer, indent=4)

    return videoRank
