from active_learning_modules import rank_by_feature
from active_learning_modules.dataset_utils import getNumFramesInVideo

labeledFeaturesPath     = "/model/work_dir/RandomCorrNetPT/Rep1/phoenix2014-T-RandomRep1-labeled-iteration1-features/train"
unlabeledFeaturesPath   = "/model/work_dir/RandomCorrNetPT/Rep1/phoenix2014-T-RandomRep1-unlabeled-iteration1-features/test"

iterId = 6

datasetBase = "phoenix2014-T"
framesInVideo = getNumFramesInVideo("phoenix2014-T")
frameBudget   = 10000
method        = 'uncast'

print(f"Running {method} on features from iteration {1}!")

similarityRank = rank_by_feature.rankSimiliratyByFeatures(  labeledFeaturesPath,
                                                        unlabeledFeaturesPath,
                                                        method,
                                                        f"./",
                                                        0,
                                                        framesInVideo,
                                                        datasetBase,
                                                        frameBudget
                                                        )

selectedVideos = {}
addedFrames    = 0
for videoId, rankingInfo in list(similarityRank.items()):
       selectedVideos[videoId] = rankingInfo

       addedFrames += framesInVideo[videoId]

       if addedFrames >= frameBudget:
              break
similarityRank = selectedVideos

print(similarityRank)

raise NotImplementedError

with open(labeledSet, "r") as f:
       labeledSet = f.readlines()
del labeledSet[0]

with open(unlabeledSet, "r") as f:
       unlabeledSet = f.readlines()
del unlabeledSet[0]


numFrames     = []
numGloss      = []
uniqueGlosses = set()
videoIdToNumGloss = {}
videoIdToGlosses  = {}
for line in labeledSet:
       videoId         = line.split("|")[0]
       
       videoIdToGlosses[videoId]   = line.split("|")[-1].split(" ")
       videoIdToNumGloss[videoId]  = len(videoIdToGlosses[videoId])

       numFrames.append(framesInVideo[videoId])
       numGloss.append(videoIdToNumGloss[videoId])
       [ uniqueGlosses.add(glosses) for glosses in videoIdToGlosses[videoId] ]

for line in unlabeledSet:
       videoId  = line.split("|")[0]
       
       videoIdToGlosses[videoId]   = line.split("|")[-1].split(" ")
       videoIdToNumGloss[videoId]  = len(videoIdToGlosses[videoId])


import numpy as np
for videoId in similarityRank.keys():
       numFrames.append(framesInVideo[videoId])
       numGloss.append(videoIdToNumGloss[videoId])

       [ uniqueGlosses.add(glosses) for glosses in videoIdToGlosses[videoId] ]

numFrames = np.sum(numFrames)
numGloss  = np.sum(numGloss)
uniqueGlosses = len(uniqueGlosses)
print(f"Labeled set on iteration {iterId+1} would look like: Total num of frames = {numFrames} over {len(similarityRank) + len(labeledSet)} videos, Total num of glosses = {numGloss}, NumUniqueGloss = {uniqueGlosses}, Gloss density = {numGloss / numFrames:.4f}, Unique gloss Density = {uniqueGlosses / numFrames:.4f}")

'''
metrics = ['n-best-bald', 
       'token-bald', 
       'token-bald-2', 
       'token-bald-4', 
       'coreset',
       'coreset-5', 
       'frame-level-bald'
       ]

for i in metrics:
    print(f"testing {i}....")
    similarityRank = rank_by_feature.rankSimiliratyByFeatures(  labeledFeaturesPath,
                                                                unlabeledFeaturesPath,
                                                                i,
                                                                "mean",
                                                                10,
                                                                f"./",
                                                                0
                                                                )
'''

