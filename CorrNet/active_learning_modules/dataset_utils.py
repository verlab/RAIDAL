import subprocess
import random
import shutil
import json
import yaml
import os


def getNumFramesInVideo(datasetBase):
    with open(f"frameCount/{datasetBase}.json", 'r') as f:
        framesInVideo = json.load(f)

    return framesInVideo


def randomlySelectVideos(unlabeledSubsetPath: str, datasetBase: str, frameBudget: int, numFramesByVideo: dict):
    if datasetBase == "phoenix2014":
        unlabeledCSV   = os.path.join(unlabeledSubsetPath,   "phoenix-2014-multisigner/annotations/manual/test.corpus.csv")
    elif datasetBase == "phoenix2014-T":
        unlabeledCSV   = os.path.join(unlabeledSubsetPath,   "annotations/manual/PHOENIX-2014-T.test.corpus.csv")
    elif datasetBase == "Isharah":
        unlabeledCSV   = os.path.join(unlabeledSubsetPath,   "Annotations/isharah1000/SI/test.txt")
    else:
        raise Exception(f"Dataset {datasetBase} in config file not recognized!")


    # Get everything in the unlabeled pool
    with open(unlabeledCSV, "r") as f:
        unlabeledPoolLines  = f.readlines()

    # Remove header
    del unlabeledPoolLines[0]

    
    random.shuffle(unlabeledPoolLines)
    
    addedFrames    = 0
    selectedVideos = {}
    for unlabeledLine in unlabeledPoolLines:
        videoId    = unlabeledLine.split("|")[0]

        selectedVideos[videoId] = 0.0

        numFrames   = numFramesByVideo[videoId]
        addedFrames += numFrames

        if addedFrames >= frameBudget:
            print(f"Added {addedFrames} out of the budget of {frameBudget} from {len(selectedVideos)} randomly selected videos.")
            break
            
    return selectedVideos



def getNumFramesInLabeledSet(labeledSubsetPath: str, datasetBase: str, numFramesByVideo: dict):
    if datasetBase == "phoenix2014":
        labeledCSV   = os.path.join(labeledSubsetPath,   "phoenix-2014-multisigner/annotations/manual/train.corpus.csv")
    elif datasetBase == "phoenix2014-T":
        labeledCSV   = os.path.join(labeledSubsetPath,   "annotations/manual/PHOENIX-2014-T.train.corpus.csv")
    elif datasetBase == "Isharah":
        labeledCSV   = os.path.join(labeledSubsetPath,   "Annotations/isharah1000/SI/train.txt")
    else:
        raise Exception(f"Dataset {datasetBase} in config file not recognized!")

    with open(labeledCSV, "r") as f:
        # Isolate the header
        labeledCSV  = f.readlines()
    
    del labeledCSV[0]

    totalNumFrames = 0
    for line in labeledCSV:
        videoId   = line.split("|")[0]
        
        if videoId not in numFramesByVideo:
            raise Exception(f"Video {videoId} not found json that counts number of frames per video!")

        numFrames = numFramesByVideo[videoId]
        totalNumFrames += numFrames

    return totalNumFrames


def preprocessRoutineWrapper(datasetPath: str, datasetName: str, datasetBase):
    """
    Runs the preprocess script to generatee the .stm groundtruth files, the gloss_dict.npy file, the [train, test, dev]_info.npy... Pretty much everything that we need to work with.
    
    It's important to note that, while the preprocessing routine in CorrNet generates the groundtruth for the train, test and dev files,
    they are not saved in the correct folder automatically. This makes it necessary to move the groundtruth files
    to their correct folder. This is what the second part of this function does

    Args:
        datasetPath (srt): The folder containing your data. It's the one with the phoenix-2014-multisigner/ folder.
        datasetName (str): The name of the dataset. Could be anything theoretically, but needs to match the one in the .yaml config file for the rest of the program to work
    """
    if datasetBase == "phoenix2014":
        subprocess.run(f"cd preprocess; python3 dataset_preprocess.py --dataset {datasetName} --dataset-root {datasetPath}/phoenix-2014-multisigner", shell=True, check=True)

        for subset in ["train", "test", "dev"]:
            currentSubsetGroundtruthPath = os.path.join(f"./preprocess/{datasetName}", f"{datasetName}-groundtruth-{subset}.stm")
            shutil.copy(currentSubsetGroundtruthPath, "./evaluation/slr_eval/")

    elif datasetBase == "phoenix2014-T":
        subprocess.run(f"cd preprocess; python3 dataset_preprocess-T.py --dataset {datasetName} --dataset-root {datasetPath}", shell=True, check=True)

        for subset in ["train", "test", "dev"]:
            currentSubsetGroundtruthPath = os.path.join(f"./preprocess/{datasetName}", f"{datasetName}-groundtruth-{subset}.stm")
            shutil.copy(currentSubsetGroundtruthPath, "./evaluation/slr_eval/")
    
    elif datasetBase == "Isharah":
        subprocess.run(f"cd preprocess; python3 dataset_preprocess-Isharah.py --dataset {datasetName} --dataset-root {datasetPath}", shell=True, check=True)

        for subset in ["train", "test", "dev"]:
            currentSubsetGroundtruthPath = os.path.join(f"./preprocess/{datasetName}", f"{datasetName}-groundtruth-{subset}.stm")
            shutil.copy(currentSubsetGroundtruthPath, "./evaluation/slr_eval/")

    else:
        raise Exception(f"Dataset {datasetBase} in config file not recognized!")


def labelDataPoints(labeledSubsetPath: str, unlabeledSubsetPath: str, selectedVideos: dict, datasetBase: str, isFirstLabelingLoop=False):
    """Moves the selected data points from the unlabeled pool to the labeled pool. 

    Since we already have all the datapoints and this is a simulation, 
    we will simply move the selected data points from the unlabeledSubset to the labeledSubset.

    More specifically, the unlabeledSubset/phoenix-2014-multisigner/annotations/manual/test.corpus.csv file is our simulated unlabeled pool.
    Over there we have a bunch of data points that the training process hasn't seen yet. The active learning loop selected which of these are the ones that the model
    wants to learn from, and this function here is responsible for moving them to the labeled pool, so the next learning round incorporates them.

    Args:
        labeledSubsetPath          (str): The path to the labeled subset
        unlabeledSubsetPath        (str): The path to the unlabeled subset
        nSamplesToLabel            (int): How many data samples should be moved to the labeled pool.
        selectedVideos            (dict): The names of the videos that were selected to be labeled. The key is the name of the video, the value is a list of the frames that should be labeled.
        isFirstLabelingLoop       (bool): Whether or not it is the first labeling loop. This will create a new labeled pool file!
    """

    if datasetBase == "phoenix2014":
        labeledCSV   = os.path.join(labeledSubsetPath,   "phoenix-2014-multisigner/annotations/manual/train.corpus.csv")
        unlabeledCSV = os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv")
    elif datasetBase == "phoenix2014-T":
        labeledCSV   = os.path.join(labeledSubsetPath,   "annotations/manual/PHOENIX-2014-T.train.corpus.csv")
        unlabeledCSV = os.path.join(unlabeledSubsetPath, "annotations/manual/PHOENIX-2014-T.test.corpus.csv")
    elif datasetBase == "Isharah":
        labeledCSV   = os.path.join(labeledSubsetPath,   "Annotations/isharah1000/SI/train.txt")
        unlabeledCSV = os.path.join(unlabeledSubsetPath, "Annotations/isharah1000/SI/test.txt")
    else:
        raise Exception(f"Dataset {datasetBase} in config file not recognized!")

    selectedVideos = list(selectedVideos.keys())

    # Get everything in the unlabeled pool
    with open(unlabeledCSV, "r") as f:
        # Isolate the header
        unlabeledPoolLines  = f.readlines()

    header = unlabeledPoolLines[0]
    del unlabeledPoolLines[0]


    # If we're on the first labeling loop, this means that our labeled pool is empty and we have to create it!
    # To do this, first create the file and add the header to it.
    if isFirstLabelingLoop:
        with open(labeledCSV, "w") as f:
            f.writelines(header) # Write the header because since this is the initial labeling, the labeledCSV file is guaranteed to be empty!
    

    # Move the videos in selectedVideos to the labeled pool and remove them from the unlabeled pool
    newlyLabeledData        = []
    remainingUnlabeledData  = unlabeledPoolLines.copy()
    
    for videoID in selectedVideos:
        for unlabeledLine in unlabeledPoolLines:
            unlabeledVideoId = unlabeledLine.split("|")[0]

            if videoID == unlabeledVideoId:
                newlyLabeledData.append(unlabeledLine)
                remainingUnlabeledData.remove(unlabeledLine)

    
    # When we 'label' an instance, we have to remove it from the unlabeled pool. It is easier to just delete the unlabeledPool file and write what we want
    # instead of figuring out what lines should be kept in or removed one by one
    with open(unlabeledCSV, "w") as f:
        f.writelines(header)
        f.writelines(remainingUnlabeledData)

    # Next, we append the selected samples to the labeledPool file and we're done!
    with open(labeledCSV, "a") as f:
        f.writelines(newlyLabeledData)


def splitDataset(datasetPath: str, customName: str, iterationId: int, datasetBase):
    """
    Creates copies of datasetPath for serving as the labeled and the unlabeled subsets in the active learning loop.
    The structure copies the one used in the phoenix2014 dataset. To avoid copying all of the data, symlinks to the original dataset are used for
    copying the images.

    Args:
        datasetPath (str): A path pointing to where the dataset is.
        customName  (str): An optional custom name for the labeled and unlabeled splits. Could be something like: 'Phoenix20AL5Steps16Frames' for example
        iterationId (str): The ID of the current AL iteration.
    """
    
    datasetParentFolder  =  "/".join(datasetPath.split("/")[ : -1])
    datasetName          =  datasetPath.split("/")[-1]
    
    # Save the path for the labeled and unlabeled folders

    labeledSubsetPath   = os.path.join(datasetParentFolder, f"{datasetName}-{customName}-labeled-iteration{iterationId}")
    unlabeledSubsetPath = os.path.join(datasetParentFolder, f"{datasetName}-{customName}-unlabeled-iteration{iterationId}")

    copyDataset(datasetPath, labeledSubsetPath,   datasetBase)
    copyDataset(datasetPath, unlabeledSubsetPath, datasetBase)

    # Save where the original annotation files are
    if datasetBase == "phoenix2014":
        trainAnnotationSuffix = "phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
        trainAnnotationPath   = os.path.join(datasetPath, trainAnnotationSuffix)

        testAnnotationSuffix  = "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"
        testAnnotationPath    = os.path.join(datasetPath, testAnnotationSuffix)

        devAnnotationSuffix   = "phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"
        devAnnotationPath     = os.path.join(datasetPath, devAnnotationSuffix)

        fullFrameFolderCategories = ["fullFrame-210x260px", "fullFrame-256x256px"]

        featuresFolderSuffix  = "phoenix-2014-multisigner/features"
        header                = "id|folder|signer|annotation"
        
    elif datasetBase == "phoenix2014-T":
        trainAnnotationSuffix = "annotations/manual/PHOENIX-2014-T.train.corpus.csv"
        trainAnnotationPath   = os.path.join(datasetPath, trainAnnotationSuffix)

        testAnnotationSuffix  = "annotations/manual/PHOENIX-2014-T.test.corpus.csv"
        testAnnotationPath    = os.path.join(datasetPath, testAnnotationSuffix)

        devAnnotationSuffix   = "annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
        devAnnotationPath     = os.path.join(datasetPath, devAnnotationSuffix)

        fullFrameFolderCategories = ["fullFrame-210x260px", "fullFrame-256x256px"]

        featuresFolderSuffix  = "features"
        header                = "name|video|start|end|speaker|orth|translation"

    elif datasetBase == "Isharah":
        trainAnnotationSuffix = "Annotations/isharah1000/SI/train.txt"
        trainAnnotationPath   = os.path.join(datasetPath, trainAnnotationSuffix)

        testAnnotationSuffix  = "Annotations/isharah1000/SI/test.txt"
        testAnnotationPath    = os.path.join(datasetPath, testAnnotationSuffix)

        devAnnotationSuffix   = "Annotations/isharah1000/SI/dev.txt"
        devAnnotationPath     = os.path.join(datasetPath, devAnnotationSuffix)

        fullFrameFolderCategories = ["fullFrame-256x256px"]

        featuresFolderSuffix  = "features"
        header                = "id|gloss"
    else:
        raise Exception(f"Dataset {datasetBase} in config file not recognized!")

    # The original train.corpus.csv annotation goes to the unlabeled subset as the test.corpus.csv file. It goes in as the test file because the test file is used for
    # running inference, and we want to run inference on the 'unlabeled' data points. Ideally it should be called something else because this gives the impression
    # that it is the original test annotation, and it might get confusing. Just keep in mind that this is our simulated unlabeled data pool, and it is only called 
    # test.corpus.csv because I don't want to change the code of CorrNet too much.

    shutil.copy(trainAnnotationPath, os.path.join(unlabeledSubsetPath, testAnnotationSuffix))
    # Since we now have a train.corpus.csv file that was 'swapped' for the test.corpus.csv file inside the unlabeled folder, we now have to also swap the
    # corresponding train and test folders inside the unlabeledSubsetPath/phoenix-2014-multisigner/features folder to reflect this change!

    # Now,    unlabeledSubsetPath/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/train
    # becomes unlabeledSubsetPath/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/test

    # And unlabeledSubsetPath/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/test
    # becomes unlabeledSubsetPath/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/train
    unlabeledSubsetFeaturesFolder = os.path.join(unlabeledSubsetPath, featuresFolderSuffix)
    for fullFrameFolderCategory in fullFrameFolderCategories:
        if datasetBase in ["phoenix2014", "phoenix2014-T"]:
            trainFullFrameCategoryFolder = os.path.join(unlabeledSubsetFeaturesFolder, f"{fullFrameFolderCategory}/train")
            testFullFrameCategoryFolder  = os.path.join(unlabeledSubsetFeaturesFolder, f"{fullFrameFolderCategory}/test")
        
            shutil.move(trainFullFrameCategoryFolder, "./tmp")
            shutil.move(testFullFrameCategoryFolder, trainFullFrameCategoryFolder)
            shutil.move("./tmp", testFullFrameCategoryFolder)

        # Note: Isharah does not require this since all videos (regardless if they are train test or dev) are inside the same "features/fullFrame-256x256px" folder!
        elif datasetBase in ["Isharah"]:
            pass
            
        else:
            raise Exception(f"Dataset {datasetBase} in config file not recognized!")

    # The training script requires a test, dev and train annotation file in every dataset, even if they are empty. Right now we only have a test file in the unlabeled set, so now we 
    # create empty dev and train files.
    with open(os.path.join(unlabeledSubsetPath, devAnnotationSuffix), "w") as outfile:
        outfile.writelines(header)
    
    with open(os.path.join(unlabeledSubsetPath, trainAnnotationSuffix), "w") as outfile:
        outfile.writelines(header)


    # Original test and dev annotations go to the labeled set as themselves. We will be testing the performance on them, so they have to go in unmodified.
    shutil.copy(testAnnotationPath, os.path.join(labeledSubsetPath, testAnnotationSuffix))
    shutil.copy(devAnnotationPath,  os.path.join(labeledSubsetPath, devAnnotationSuffix))


    return labeledSubsetPath, unlabeledSubsetPath


def copyDataset(datasetPath, newDatasetPath, datasetBase, copyAnnotations=False) -> None:
    """
    Originally meant as a part of the splitDataset function, but became more modular to allow creating copies of everything. 
    Now, you just pass a name and it will create a copy of the original with that name.

    The structure copies the one used in the phoenix2014 dataset. To avoid copying all of the data, symlinks to the original dataset are used for
    copying the images. 
    
    The main difference from just using cp -r is that cp -r would make a copy of everything, and this function smartly uses symlinks in larger folders
    with images to avoid making unecessary copies.

    Args:
        datasetPath      (str): a path pointing to where the dataset is
        newDatasetPath   (str): Where the copy is going to be created
        copyAnnotations (bool): Whether to copy the original annotation files to the new directory as well. Usually not recommended because
                                it might mess things up when splitDataset() tries to set up the labeled and unlabeled pools. But if what
                                you want is simply to make a perfect copy of datasetPath, then you can enable it without problems!
    """

    if datasetBase == "phoenix2014":
        # These will all be created by the script automatically. After they're created, we will create symlinks to the relevant files and folders.
        internalFolders = [
                            "phoenix-2014-multisigner/annotations/manual",
                            "phoenix-2014-multisigner/evaluation",
                            "phoenix-2014-multisigner/features/fullFrame-210x260px",
                            "phoenix-2014-multisigner/features/fullFrame-256x256px"
                            ]
        
        datasetSplits = [ "phoenix-2014-multisigner/annotations/manual/train.corpus.csv",
                          "phoenix-2014-multisigner/annotations/manual/test.corpus.csv",
                          "phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"]

        fullFrameFolderCategories = ["fullFrame-210x260px", "fullFrame-256x256px"]

        featuresPrefix = "phoenix-2014-multisigner/features/"
        lmdbDataPrefix = "phoenix-2014-multisigner/features/phoenix2014_videos.lmdb"

    elif datasetBase == "phoenix2014-T":
        internalFolders = [
                            "annotations/manual",
                            "evaluation",
                            "features/fullFrame-210x260px",
                            "features/fullFrame-256x256px"
                            ]

        datasetSplits = ["annotations/manual/PHOENIX-2014-T.train.corpus.csv",
                         "annotations/manual/PHOENIX-2014-T.test.corpus.csv",
                         "annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
                         ]

        fullFrameFolderCategories = ["fullFrame-210x260px", "fullFrame-256x256px"]

        featuresPrefix = "features/"
        lmdbDataPrefix = "features/phoenix2014-T_videos.lmdb"
    
    elif datasetBase == "Isharah":
        internalFolders = [
                            "Annotations/isharah1000/SI",
                            "features/"
                            ]

        datasetSplits = ["Annotations/isharah1000/SI/train.txt",
                         "Annotations/isharah1000/SI/test.txt",
                         "Annotations/isharah1000/SI/dev.txt"
                         ]
        
        fullFrameFolderCategories = ["fullFrame-256x256px"]

        featuresPrefix = "features/"
        lmdbDataPrefix = "features/Isharah_videos.lmdb"

    else:
        raise Exception(f"Dataset {datasetBase} in config file not recognized!")

    for folder in internalFolders:
        os.makedirs(os.path.join(newDatasetPath, folder))

    
    # Creates symlinks {datasetPath}/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/[train, test, dev] -> 
    # -> {newDatasetPath}/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/[train, test, dev]
    for fullFrameFolderCategory in fullFrameFolderCategories:
        originalFullFrameFolder = os.path.join(datasetPath, featuresPrefix, fullFrameFolderCategory)
        newFullFrameFolder      = os.path.join(newDatasetPath, featuresPrefix, fullFrameFolderCategory)
        
        if datasetBase in ["phoenix2014", "phoenix2014-T"]:
            for subset in ["train", "test", "dev"]:
                originalSubsetPath = os.path.join(originalFullFrameFolder, subset)
                newSubsetPath      = os.path.join(newFullFrameFolder, subset)

                os.symlink(originalSubsetPath, newSubsetPath)

        elif datasetBase in ["Isharah"]:
            os.symlink(originalFullFrameFolder, newFullFrameFolder)
        
        else:
            raise Exception(f"Dataset {datasetBase} in config file not recognized!")


    if os.path.exists(os.path.join(datasetPath,    lmdbDataPrefix)):
        originalLmdb = os.path.join(datasetPath,    lmdbDataPrefix)
        newLmdb      = os.path.join(newDatasetPath, lmdbDataPrefix)

        os.symlink(originalLmdb, newLmdb)


    # Copies {datasetPath}/phoenix-2014-multisigner/annotations/manual/[train,test,dev].corpus.csv -> {newDatasetPath}/phoenix-2014-multisigner/annotations/manual/[train,test,dev].corpus.csv
    if copyAnnotations:
        for split in datasetSplits:
            originalAnnotationPath   = os.path.join(datasetPath,    split)
            newDatasetAnnotationPath = os.path.join(newDatasetPath, split)
            
            shutil.copy(originalAnnotationPath, newDatasetAnnotationPath)



def copySetsFromCheckpoint(checkpointPath, labeledSubsetPath, unlabeledSubsetPath, datasetBase, configFile):
    """
    Copies the labeled and unlabeled sets from the checkpoint to the subset
    """
    if datasetBase == "phoenix2014":
        shutil.copy(os.path.join(checkpointPath, "labeled.corpus.csv"),
                    os.path.join(labeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/train.corpus.csv")
                    )
        
        # Copy the unlabeled checkpoint to the unlabeled subset
        shutil.copy(os.path.join(checkpointPath, "unlabeled.corpus.csv" ),
                    os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv")
                    )
    elif datasetBase == "phoenix2014-T":
        shutil.copy(os.path.join(checkpointPath, "labeled.corpus.csv"),
                    os.path.join(labeledSubsetPath, "annotations/manual/PHOENIX-2014-T.train.corpus.csv")
                    )
        
        # Copy the unlabeled checkpoint to the unlabeled subset
        shutil.copy(os.path.join(checkpointPath, "unlabeled.corpus.csv" ),
                    os.path.join(unlabeledSubsetPath, "annotations/manual/PHOENIX-2014-T.test.corpus.csv")
                    )

    elif datasetBase == "Isharah":
        shutil.copy(os.path.join(checkpointPath, "labeled.corpus.csv"),
                    os.path.join(labeledSubsetPath, "Annotations/isharah1000/SI/train.txt")
                    )
        
        # Copy the unlabeled checkpoint to the unlabeled subset
        shutil.copy(os.path.join(checkpointPath, "unlabeled.corpus.csv" ),
                    os.path.join(unlabeledSubsetPath, "Annotations/isharah1000/SI/test.txt")
                    )

    else:
        raise Exception(f"Dataset {datasetBase} in config file {configFile} not recognized!")


def copySetsToWorkDir(workDirPath, labeledSubsetName, labeledSubsetPath, unlabeledSubsetPath, datasetBase, configFile):
    # Copy the labeled and unlabeled pools as a CSV to the workdir. They will serve as a checkpoint.
    if datasetBase == "phoenix2014":
        shutil.copy(os.path.join(labeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/train.corpus.csv"),
                    os.path.join(workDirPath, labeledSubsetName, "labeled.corpus.csv")
                    )
        shutil.copy(os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"),
                    os.path.join(workDirPath, labeledSubsetName, "unlabeled.corpus.csv")
                    )
    elif datasetBase == "phoenix2014-T":
        shutil.copy(os.path.join(labeledSubsetPath, "annotations/manual/PHOENIX-2014-T.train.corpus.csv"),
                    os.path.join(workDirPath, labeledSubsetName, "labeled.corpus.csv")
                    )
        shutil.copy(os.path.join(unlabeledSubsetPath, "annotations/manual/PHOENIX-2014-T.test.corpus.csv"),
                    os.path.join(workDirPath, labeledSubsetName, "unlabeled.corpus.csv")
                    )

    elif datasetBase == "Isharah":
        shutil.copy(os.path.join(labeledSubsetPath, "Annotations/isharah1000/SI/train.txt"),
                    os.path.join(workDirPath, labeledSubsetName, "labeled.corpus.csv")
                    )
        shutil.copy(os.path.join(unlabeledSubsetPath, "Annotations/isharah1000/SI/test.txt"),
                    os.path.join(workDirPath, labeledSubsetName, "unlabeled.corpus.csv")
                    )

    else:
        raise Exception(f"Dataset {datasetBase} in config file {configFile} not recognized!")


def createConfigFiles(labeledSubsetPath, unlabeledSubsetPath, labeledSubsetName, unlabeledSubsetName, datasetBase, configFile):
    for subsetPath, subsetName in zip([labeledSubsetPath, unlabeledSubsetPath], [labeledSubsetName, unlabeledSubsetName]):
        if datasetBase == "phoenix2014":
            config = dict(dataset_root      = f"{subsetPath}/phoenix-2014-multisigner",
                        dict_path         =  './preprocess/phoenix2014/gloss_dict.npy',
                        evaluation_dir    =  './evaluation/slr_eval',
                        evaluation_prefix = f"{subsetName}-groundtruth"
                        )
        elif datasetBase == "phoenix2014-T":
            config = dict(dataset_root      = f"{subsetPath}",
                        dict_path         =  './preprocess/phoenix2014-T/gloss_dict.npy',
                        evaluation_dir    =  './evaluation/slr_eval',
                        evaluation_prefix = f"{subsetName}-groundtruth"
                        )
        elif datasetBase == "Isharah":
            config = dict(dataset_root      = f"{subsetPath}",
                        dict_path         =  './preprocess/Isharah/gloss_dict.npy',
                        evaluation_dir    =  './evaluation/slr_eval',
                        evaluation_prefix = f"{subsetName}-groundtruth"
                        )
        else:
            raise Exception(f"Dataset {datasetBase} in config file {configFile} not recognized!")

        with open(f"./configs/{subsetName}.yaml", "w") as outfile:
            yaml.dump(config, outfile)
