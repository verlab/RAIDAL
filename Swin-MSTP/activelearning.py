# Separar os dados de treino em 2 subconjuntos: treino unlabeled e treino labeled

# Passo 1:
    # Treinar o main.py com o conjunto de treino labeled

# Passo 2:
    # Salvar o melhor modelo

# Passo 3:
    # Rodar inferência do modelo no conjunto de treino unlabeled e add as 10% mais incertas ao conjunto de treino labeled

# Passo 4:
    # Voltar ao passo 1 e repetir até acabar as amostras

import numpy as np
import subprocess
import shutil
import random
import yaml
import json
import os
import re

from active_learning_modules.dataset_utils import splitDataset, copyDataset, labelDataPoints, preprocessRoutineWrapper, getNumFramesInLabeledSet, randomlySelectVideos, getNumFramesInVideo, copySetsFromCheckpoint, copySetsToWorkDir, createConfigFiles
from active_learning_modules.parser import makeParser, validateParams, saveALParams
from active_learning_modules.rank_by_feature import rankSimiliratyByFeatures


if __name__ == '__main__':
    args = makeParser().parse_args()
    validateParams(args)

    with open(args.config, "r") as f:
        c           = yaml.safe_load(f)
        seed        = c['random_seed']
        datasetBase = c['dataset'] # Must ALWAYS be one of ['phoenix2014', 'phoenix2014-T', 'Isharah']
        if datasetBase not in ['phoenix2014', 'phoenix2014-T', 'Isharah']:
            raise Exception(f"Dataset \'{datasetBase}\' in config file {args.config} not recognized!")

    random.seed(seed)
    np.random.seed(seed)

    modelNeedsTraining     = True
    needsFeatureGeneration = True


    # Load the dict that tells me how many frames each video has
    framesInVideo = getNumFramesInVideo(datasetBase)

    # if not loading a checkpoint
    if args.checkpoint is None:
        iterationId = 1

        # Creates a labeled and unlabeled subset of the dataset. 
        # They are named {datasetParentFolder}/{datasetName}-[labeled, unlabeled]-iteration{iterationId}
        labeledSubsetPath, unlabeledSubsetPath = splitDataset(args.dataset_path, args.custom_name, iterationId, datasetBase)

        labeledSubsetName   = labeledSubsetPath.split("/")[-1]
        unlabeledSubsetName = unlabeledSubsetPath.split("/")[-1]
        
        selectedVideos = randomlySelectVideos(  unlabeledSubsetPath, 
                                                datasetBase, 
                                                args.num_frames_to_label[0],
                                                framesInVideo
                                                )

        # Now we do our first labeling loop. This will create a train.corpus.csv file for the labeled subset.
        labelDataPoints(labeledSubsetPath,
                        unlabeledSubsetPath,
                        selectedVideos,
                        datasetBase,
                        isFirstLabelingLoop=True
                        )
        
    # If loading from a checkpoint
    else:
        # Find the iterationId of the checkpoint
        iterationId    = int(re.findall("(\d+)", args.checkpoint.split("-iteration")[-1])[-1])

        labeledSubsetPath, unlabeledSubsetPath = splitDataset(args.dataset_path, 
                                                              args.custom_name, 
                                                              iterationId,
                                                              datasetBase
                                                              )

        labeledSubsetName   = labeledSubsetPath.split("/")[-1]
        unlabeledSubsetName = unlabeledSubsetPath.split("/")[-1]

        # Create the work_dir folders for the checkpoint
        os.makedirs(os.path.join(args.work_dir, labeledSubsetName))
        os.makedirs(os.path.join(args.work_dir, unlabeledSubsetName))


        # Copy the labeled and unlabeled pools from the checkpoint to the labeled and unlabeled subsets
        copySetsFromCheckpoint(args.checkpoint, labeledSubsetPath, unlabeledSubsetPath, datasetBase, args.config)

        # If true, uses the pretrained model found in the checkpoint to generate the features.
        # Setting 'modelNeedsTraining' avoids retraining from scratch and makes comparisons more fair.
        if args.use_model:
            # Copy the trained model to the labeled checkpoint
            shutil.copy(os.path.join(args.checkpoint, "_best_model.pt"),
                        os.path.join(args.work_dir, labeledSubsetName, "_best_model.pt"))
            
            modelNeedsTraining = False
        
        # Alternatively, we can use pregenerated features to make sure every AL method
        # gets the same features.
        if args.use_features is not None:

            # Copy the trained model to the labeled checkpoint
            shutil.copy(os.path.join(args.checkpoint, "_best_model.pt"),
                        os.path.join(args.work_dir, labeledSubsetName, "_best_model.pt"))
            
            # Copy the test and dev results
            shutil.copy(os.path.join(args.checkpoint, "test.txt"),
                        os.path.join(args.work_dir, labeledSubsetName, "test.txt"))
            
            shutil.copy(os.path.join(args.checkpoint, "dev.txt"),
                        os.path.join(args.work_dir, labeledSubsetName, "dev.txt"))
            

            labeledFeaturesPath   = os.path.join(args.use_features[0], "train")
            unlabeledFeaturesPath = os.path.join(args.use_features[1], "test")

            modelNeedsTraining     = False
            needsFeatureGeneration = False
    
    if args.checkpoint is None:
        idxFrameBudget  = 1
        budgetOverflow  = getNumFramesInLabeledSet(labeledSubsetPath, datasetBase, framesInVideo) - args.num_frames_to_label[0]
        extraIterations = 0
    else:
        idxFrameBudget  = 0
        budgetOverflow  = 0
        extraIterations = 1

    for iterationId in range(iterationId, iterationId + args.num_iterations + extraIterations):
        framesInLabeledSet = getNumFramesInLabeledSet(labeledSubsetPath, datasetBase, framesInVideo)

        print(f"Starting AL iteration {iterationId} now....")
        print(f"Labeled set has {framesInLabeledSet} frames! Budget overflow is {budgetOverflow}")
        
        frameBudget    = args.num_frames_to_label[idxFrameBudget] - budgetOverflow
        idxFrameBudget = min(len(args.num_frames_to_label)-1, idxFrameBudget+1) # Limit value for the last iteration.

        # Create config files for the unlabeled and labeled datasets
        createConfigFiles(labeledSubsetPath, unlabeledSubsetPath, labeledSubsetName, unlabeledSubsetName, datasetBase, args.config)
        
        # Run preprocess routine for labeled subset
        preprocessRoutineWrapper(labeledSubsetPath, labeledSubsetName, datasetBase)

        # Train video2gloss model on labeled subset
        if modelNeedsTraining:
            subprocess.run(f"python3 main.py \
                            --config {args.config} \
                            --device {args.device} \
                            --dataset {labeledSubsetName} \
                            --work-dir {args.work_dir}/{labeledSubsetName}/", 
                            shell=True, 
                            check=True)
        
        # Copy the labeled and unlabeled pools as a CSV to the workdir. They will serve as a checkpoint.
        copySetsToWorkDir(args.work_dir, labeledSubsetName, labeledSubsetPath, unlabeledSubsetPath, datasetBase, args.config)
        
        # Delete models with non-optimal training weights
        subprocess.run(f"rm {args.work_dir}/{labeledSubsetName}/dev*.pt", shell=True)
        
        subprocess.run(f"./cleanup.sh {args.work_dir}/{labeledSubsetName}", shell=True)

        # Now, we start the part of the Active Learning loop where we look for significant samples in the unlabeled subset.
        # Run preprocess routine for the unlabeled subset
        preprocessRoutineWrapper(unlabeledSubsetPath, unlabeledSubsetName, datasetBase)

        # If the selection strategy is randomly sampling videos, we randomly select a couple of videos for the labeled pool and that's it!
        if args.strategy == "random":
            if iterationId == 1:
                subprocess.run(f"python main.py --config {args.config} --device {args.device} --dataset {labeledSubsetName}   --phase features --load-weights {args.work_dir}/{labeledSubsetName}/_best_model.pt --work-dir {args.work_dir}/{labeledSubsetName}-features/   --feature-folders train --batch-size 1 --test-batch-size 1 -b {args.bald_iterations}", shell=True, check=True)
                subprocess.run(f"python main.py --config {args.config} --device {args.device} --dataset {unlabeledSubsetName} --phase features --load-weights {args.work_dir}/{labeledSubsetName}/_best_model.pt --work-dir {args.work_dir}/{unlabeledSubsetName}-features/ --feature-folders test  --batch-size 1 --test-batch-size 1 --test-inference -b {args.bald_iterations}", shell=True, check=True)
            
            videoRank = randomlySelectVideos(unlabeledSubsetPath,
                                                  datasetBase,
                                                  frameBudget,
                                                  framesInVideo
                                                  )
            
        else:
            # Save the AL parameters in the work_dir
            saveALParams(args, os.path.join(args.work_dir, labeledSubsetName))

            # Now we use our SlowFastModel that was trained on the labeledSubset to extract features from all the data in the 
            # labeled and unlabeled subsets.

            if needsFeatureGeneration:
                subprocess.run(f"python main.py --config {args.config} --device {args.device} --dataset {labeledSubsetName}   --phase features --load-weights {args.work_dir}/{labeledSubsetName}/_best_model.pt --work-dir {args.work_dir}/{labeledSubsetName}-features/   --feature-folders train --batch-size 1 --test-batch-size 1 -b {args.bald_iterations}", shell=True, check=True)
                subprocess.run(f"python main.py --config {args.config} --device {args.device} --dataset {unlabeledSubsetName} --phase features --load-weights {args.work_dir}/{labeledSubsetName}/_best_model.pt --work-dir {args.work_dir}/{unlabeledSubsetName}-features/ --feature-folders test  --batch-size 1 --test-batch-size 1 --test-inference -b {args.bald_iterations}", shell=True, check=True)
                
                # These point to the folder containing the extracted features for each video
                labeledFeaturesPath   = os.path.join(f"{args.work_dir}/{labeledSubsetName}-features",   "train")
                unlabeledFeaturesPath = os.path.join(f"{args.work_dir}/{unlabeledSubsetName}-features", "test")

            # Use the features to find the data points in the unlabeled set that are the most dissimilar to the ones in the labeled set.
            videoRank = rankSimiliratyByFeatures(  labeledFeaturesPath, 
                                                        unlabeledFeaturesPath,
                                                        args.strategy,
                                                        f"{args.work_dir}/{labeledSubsetName}/",
                                                        args.device,
                                                        framesInVideo,
                                                        datasetBase,
                                                        frameBudget
                                                        )
        
        selectedVideos = {}
        addedFrames    = 0
        for videoId, rankingInfo in videoRank.items():
            selectedVideos[videoId] = rankingInfo
            
            addedFrames += framesInVideo[videoId]

            if addedFrames >= frameBudget:
                budgetOverflow = addedFrames - frameBudget
                break


        # Prepare subset path for the next run
        newLabeledSubsetPath   = labeledSubsetPath.rstrip(str(iterationId))   + str(iterationId+1)
        newUnlabeledSubsetPath = unlabeledSubsetPath.rstrip(str(iterationId)) + str(iterationId+1)

        # Copy the data in the current dataset to the dataset of the next run
        copyDataset(labeledSubsetPath,   
                    newLabeledSubsetPath,
                    datasetBase,
                    copyAnnotations=True
                    )
                    
        copyDataset(unlabeledSubsetPath, 
                    newUnlabeledSubsetPath,
                    datasetBase,
                    copyAnnotations=True)

        # Update unlabeledSubset[Name, Path] and labeledSubset[Name, Path]
        labeledSubsetPath   = newLabeledSubsetPath
        labeledSubsetName   = labeledSubsetPath.split("/")[-1]
        unlabeledSubsetPath = newUnlabeledSubsetPath
        unlabeledSubsetName = unlabeledSubsetPath.split("/")[-1]

        # Update flag so a model is trained in the next AL loop
        modelNeedsTraining     = True
        # Update flag so the inference pass for the unlabeled set is run 
        # in the next AL loop
        needsFeatureGeneration = True

        # Add the samples in selectedVideos to the labeled set.
        labelDataPoints(labeledSubsetPath, 
                        unlabeledSubsetPath, 
                        selectedVideos,
                        datasetBase,
                        isFirstLabelingLoop=False
                        )
