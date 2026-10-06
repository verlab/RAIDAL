import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

def normalizeEmbeddings(feature_dict, device):
    """L2-normalizes each video's per-frame features, then averages them into one
    global, video-level embedding.

    This is a local variant of `active_learning_modules.utils.normalizeEmbeddings`
    (kept separate on purpose, only shadowing that name within this module):
    global-pooled Core-set needs a single vector per video rather than a
    per-frame sequence, so the extra `torch.mean` collapses the temporal
    dimension right after normalization.

    Args:
        feature_dict (dict): Maps video id -> array/tensor of shape (T, D), the
            per-frame feature vectors for that video (T frames, D-dim embeddings).
        device (str): The torch device used to run the normalization.

    Returns:
        dict: Maps video id -> a single numpy array of shape (D,), the mean of
            that video's L2-normalized per-frame embeddings, stored on the CPU.
    """
    normDict = {}
    for vid, embeddings in feature_dict.items():
        # Pass to GPU and normalize per-frame
        embeddings = torch.as_tensor(embeddings).float().to(device)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Average them
        embeddings = torch.mean(embeddings, axis=0)

        # Send back to CPU to avoid breaking the coreset function.
        normDict[vid] = embeddings.cpu().numpy() 
    
    return normDict


def globalPooledCoreset(unlabeledSet: dict, labeledSet: dict, device: str, framesInVideo=None, frameBudget=None) -> dict:
    """Runs the "Core-set" baseline: greedy selection over globally-pooled videos.

    This is the plain Core-set baseline from Sec. 2.2 / 4.2.1: each video is
    collapsed into a single globally-pooled embedding (see `normalizeEmbeddings`
    above), and the greedy k-center selection in `coreset` operates directly on
    these per-video vectors. Unlike RAIDAL and Soft-Attention Core-set, there is
    no per-frame or TROI-filtered scoring: the whole video is one point in
    feature space.

    Args:
        unlabeledSet (dict): Maps video id -> dict with at least the key
            'biLSTMFeaturesNoDropout' (per-frame features, shape (T, D)), for
            every candidate video.
        labeledSet (dict): Same structure as `unlabeledSet`, but for videos
            already in the labeled pool. Pooled the same way and used in full as
            the reference set.
        device (str): The torch device used for all tensor operations.
        framesInVideo (dict, optional): Maps video id -> raw frame count T_i.
            Only used to track annotation cost against `frameBudget`; unlike
            RAIDAL and Soft-Attention Core-set, the acquisition score itself is
            not normalized by frame count here.
        frameBudget (int, optional): Total number of raw frames to select before
            the greedy selection in `coreset` stops. If either this or
            `framesInVideo` is None, every video is ranked instead.

    Returns:
        dict: Maps video id -> acquisition score (the distance to the nearest
            labeled or previously-selected video), in the order videos were
            selected (most informative first).
    """
    # We use biLSTMFeaturesNoDropout, since they come from the bi-LSTM layer and
    # are rich in temporal context and sign trajectory information.
    labeledEmbeddingsBiLSTM   = {k: v['biLSTMFeaturesNoDropout'] for k, v in labeledSet.items()}
    unlabeledEmbeddingsBiLSTM = {k: v['biLSTMFeaturesNoDropout'] for k, v in unlabeledSet.items()}
    labeledEmbeddingsBiLSTM   = normalizeEmbeddings(labeledEmbeddingsBiLSTM,   device)
    unlabeledEmbeddingsBiLSTM = normalizeEmbeddings(unlabeledEmbeddingsBiLSTM, device)
    
    scoresBiLSTM = coreset(unlabeledEmbeddingsBiLSTM, labeledEmbeddingsBiLSTM, device, framesInVideo, frameBudget)

    return scoresBiLSTM



def coreset(unlabeledFeatures: dict, labeledFeatures: dict, device: str, framesInVideo=None, frameBudget=None) -> dict:
    """Greedily selects the globally-pooled video farthest from the labeled pool.

    Vectorized greedy k-center / farthest-point selection: instead of looping
    over Python dicts like `raidal.coreset` does, every video's single pooled
    embedding is stacked into one (numVideos, D) matrix, so the distance to the
    labeled pool -- and the running per-video minimum -- can be updated for every
    remaining video at once.

    Args:
        unlabeledFeatures (dict): Maps video id -> a single globally-pooled
            embedding (shape (D,)) for every candidate video.
        labeledFeatures (dict): Maps video id -> a single globally-pooled
            embedding for every video already in the labeled pool.
        device (str): The torch device used for all tensor operations.
        framesInVideo (dict, optional): Maps video id -> raw frame count T_i.
            Only consulted, together with `frameBudget`, to decide when to stop
            selecting; the score itself is not normalized by it.
        frameBudget (int, optional): Total number of raw frames to select before
            stopping. If this or `framesInVideo` is None, every video is ranked.

    Returns:
        dict: Maps video id -> the distance to the nearest labeled (or
            already-selected) video, in the order videos were selected (most
            informative first).
    """
    # videoRank is where we store the score of each video. Since Python dicts
    # preserve insertion order, iterating videoRank later also recovers the
    # acquisition order (most informative video first).
    videoRank        = {}

    # torch has no notion of a "video id" index, so idxToName lets us map a row
    # index in the stacked feature matrices below back to the video id it
    # came from.
    idxToName = {}
    idx       = 0
    for videoId in unlabeledFeatures.keys():
        idxToName[idx] = videoId
        idx += 1

    # These are the video features bundled up in an array of shape (numberOfVideos, D).
    unlabeledVideoFeatures = torch.as_tensor(np.array([videoFeature for videoFeature in unlabeledFeatures.values()])).float().to(device)
    labeledVideoFeatures   = torch.as_tensor(np.array([videoFeature for videoFeature in labeledFeatures.values()  ])).float().to(device)
    
    # Calculates the eucledian distance between two N-dimensional vectors using torch.
    # vec1[None] adds a batch dimension (cdist expects 2D inputs on both sides),
    # and the trailing [0] then drops that same dimension from the result, leaving
    # a plain 1D vector of distances from vec1 to every row of vec2.
    distanceFunction = lambda vec1, vec2: torch.cdist(vec1[None], vec2)[0]

    minDistances = torch.full(size=(unlabeledVideoFeatures.shape[0],), 
                                    fill_value=float('inf')
                                    ).to(device)


    # Calculate the minimum distance from every unlabeled video to the labeled set.
    for videoFeature in tqdm(labeledVideoFeatures, unit="videos", desc="Calculating initial distance to labeled set"):
        dist            = distanceFunction(videoFeature, unlabeledVideoFeatures)
        minDistances    = torch.minimum(minDistances, dist)


    addedFrames = 0

    # Uses the precomputed distances to the labeled set (minDistances) to select the video with the farthest embeddings and store
    # its "difference" value in minDistances.
    # Then, recomputes the distances of the other unlabeled video to the picked video and
    # updates when the new distance is lower than the previously stored distance.
    # Repeat until all videos have been scored;
    for i in tqdm(range(unlabeledVideoFeatures.shape[0]), desc="Ranking videos", unit="video"):
        selectedVideoId = torch.argmax(minDistances)
        selectedVideoId = int(selectedVideoId.cpu().item())

        # Update the rank of the selected video by adding the value in minDistances
        selectedVideoName = idxToName[selectedVideoId]
        videoRank[selectedVideoName] = minDistances[selectedVideoId].item()

        # Update the distance of the selected video in minDistances to avoid it getting picked again.
        # I chose to update the mask value to -inf, because this way we guarantee that it will never get picked again
        minDistances[selectedVideoId] = float("-inf")
        
        selectedVideoFeature = unlabeledVideoFeatures[selectedVideoId]

        # Update the distance of the other unlabeled videos to be the minimum between their original value
        # and the distance to the selected video
        dist          = distanceFunction(selectedVideoFeature, unlabeledVideoFeatures)
        minDistances  = torch.minimum(minDistances, dist)

        if framesInVideo is not None and frameBudget is not None:
            numFrames     = framesInVideo[selectedVideoName]
            addedFrames  += numFrames
            if addedFrames >= frameBudget:
                print(f"Added {addedFrames} out of the budget of {frameBudget}. Stopping....")
                break

    unlabeledVideoFeatures = unlabeledVideoFeatures.to("cpu")
    labeledVideoFeatures   = labeledVideoFeatures.to("cpu")
    minDistances           = minDistances.to("cpu")

    del (unlabeledVideoFeatures, labeledVideoFeatures, minDistances)

    with torch.cuda.device(device):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return videoRank