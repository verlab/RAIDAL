import torch
import numpy as np

from tqdm import tqdm

from active_learning_modules.utils import normalizeEmbeddings


def unfilteredRaidal(unlabeledSet: dict, labeledSet: dict, device: str, framesInVideo: dict, frameBudget=None) -> dict:
    """Runs RAIDAL's Information Density scoring without the TROI filtering stage.

    This is the "RAIDAL (No Filter)" ablation from Sec. 5.4 / Table 6(a): it reuses
    the same greedy Core-set scoring as RAIDAL (Eq. 4), but skips
    Redundancy-Aware Filtering entirely. No CTC alignment peaks are extracted and
    no TROI is built, so every timestep of every video -- rest poses, pauses, and
    redundant frames included -- contributes to the acquisition score.

    Args:
        unlabeledSet (dict): Maps video id -> dict with at least the key
            'biLSTMFeaturesNoDropout' (per-frame features, shape (T, D)), for
            every candidate video.
        labeledSet (dict): Same structure as `unlabeledSet`, but for videos already
            in the labeled pool. Used in full as the reference set F_L in Eq. 4.
        device (str): The torch device used for all tensor operations.
        framesInVideo (dict): Maps video id -> raw frame count T_i, used to
            normalize the acquisition score by annotation cost (Eq. 4).
        frameBudget (int, optional): Total number of raw frames to select across
            the unlabeled pool before the greedy selection in `coreset` stops.

    Returns:
        dict: Maps video id -> acquisition score, in the order videos were
            selected by the greedy Core-set procedure (most informative first).
    """
    # We use biLSTMFeaturesNoDropout, since they come from the bi-LSTM layer and
    # are rich in temporal context and sign trajectory information.
    labeledEmbeddingsBiLSTM   = {k: v['biLSTMFeaturesNoDropout'] for k, v in labeledSet.items()}
    unlabeledEmbeddingsBiLSTM = {k: v['biLSTMFeaturesNoDropout'] for k, v in unlabeledSet.items()}
    labeledEmbeddingsBiLSTM   = normalizeEmbeddings(labeledEmbeddingsBiLSTM,   device)
    unlabeledEmbeddingsBiLSTM = normalizeEmbeddings(unlabeledEmbeddingsBiLSTM, device)
    
    scoresBiLSTM = coreset(unlabeledEmbeddingsBiLSTM, labeledEmbeddingsBiLSTM, device, framesInVideo, frameBudget)

    return scoresBiLSTM


def coreset(unlabeledSet: dict, labeledSet: dict, device: str, framesInVideo: dict, frameBudget=None) -> dict:
    """Greedily ranks unlabeled videos by Information Density over their full features.

    Same greedy farthest-point procedure as `raidal.coreset` (Sec. 3.2.2, Eq. 4),
    but with no TROI filtering applied beforehand: `frameFeatures` below holds
    every frame of the video, not a decoder-aligned subset.

    Args:
        unlabeledSet (dict): Maps video id -> L2-normalized per-frame embeddings
            (shape (T, D)) for every candidate video.
        labeledSet (dict): Maps video id -> L2-normalized per-frame embeddings for
            every video already in the labeled pool. Used in full as the
            reference set F_L in Eq. 4.
        device (str): The torch device used for all tensor operations.
        framesInVideo (dict): Maps video id -> raw frame count T_i, used both to
            normalize the acquisition score (Eq. 4) and to track annotation cost
            against `frameBudget`.
        frameBudget (int, optional): Total number of raw frames to select before
            stopping.

    Returns:
        dict: Maps video id -> acquisition score (Eq. 4), in the order videos were
            selected (most informative first).
    """
    unlabeledData  = {}
    for videoId, embeddings in tqdm(unlabeledSet.items(), desc="Loading videos..."):
        unlabeledData[videoId] = torch.as_tensor(embeddings).float().to(device)


    labeledList         = [f for v in labeledSet.values() for f in v]
    labeledPoolFeatures = torch.as_tensor(np.array(labeledList)).float().to(device)

    # videoRank is where we store the score of each video. Since Python dicts
    # preserve insertion order, iterating videoRank later also recovers the
    # acquisition order (most informative video first); the score itself is kept
    # mainly for logging/inspection.
    videoRank        = {}
    availableVideos  = set(unlabeledData.keys())

    minDistToLabeled = {}
    
    # Calculate the minimum distance from every unlabeled sample to the labeled
    # set. minDistToLabeled[videoId] holds, per frame of that video, the distance
    # to its nearest neighbor currently in the labeled pool -- the inner
    # min_{f_l in F_L} term of Eq. 4, computed once up front for every video.
    for videoId in availableVideos:
        frameFeatures             = unlabeledData[videoId]
        dists                     = torch.cdist(frameFeatures, labeledPoolFeatures)
        minDist, _                = dists.min(dim=1)
        minDistToLabeled[videoId] = minDist

    # We don't need the labeled pool anymore. We can safely delete it from memory.
    del labeledPoolFeatures
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    addedFrames = 0

    # For each video, sums the per-frame minimum distances and normalizes by its
    # raw frame count (Eq. 4) to get its score; the video with the highest score
    # is selected.
    for _ in tqdm(range(len(unlabeledData)), desc="Ranking videos", unit="video"):
        bestVideoId    = None
        bestVideoScore = -1.0

        # Calculates the score of each video to find the best one
        for videoId in availableVideos:
            dists = minDistToLabeled[videoId]
            totalDiversity = torch.sum(dists).item()
            numFrames      = framesInVideo[videoId]

            score = totalDiversity / numFrames
            
            if score > bestVideoScore:
                bestVideoScore     = score
                bestVideoId        = videoId
                numFramesBestVideo = numFrames  # stays set to the winner's frame
                                                 # count after the loop ends: Python
                                                 # scopes for-loop variables to the
                                                 # enclosing function, not to the
                                                 # loop body itself.

        if bestVideoId is None:
            raise Exception("There's no best video! This shouldn't happen!")
        
        # Store the rank of the best video and remove it from the available pool
        videoRank[bestVideoId] = bestVideoScore
        availableVideos.remove(bestVideoId)

        # Update minDistances to the minimum distance between the original stored value and the distance
        # to the newly labeled video.
        # Core-set's greedy update: instead of recomputing every remaining
        # video's distance against the whole, now-larger labeled pool, we only
        # compare against the video we just added and keep the running minimum --
        # a point added earlier can't suddenly become a closer neighbor, so the
        # previous minimum is still valid wherever it was already smaller.
        newLabeledVideoEmbeddings = unlabeledData[bestVideoId]
        for videoId in availableVideos:
            videoEmbeddings = unlabeledData[videoId]

            distsToNew  = torch.cdist(videoEmbeddings, newLabeledVideoEmbeddings)
            minToNew, _ = distsToNew.min(dim=1)

            minDistToLabeled[videoId] = torch.minimum(minDistToLabeled[videoId], minToNew)

        addedFrames += numFramesBestVideo
        if addedFrames >= frameBudget:
            print(f"Added {addedFrames} out of the budget of {frameBudget}. Stopping....")
            break

    # Drop each video's tensors before deleting the containing dicts, so the GPU
    # memory backing them is freed here rather than left dangling until Python's
    # garbage collector gets around to it.
    for vid in unlabeledData:
        unlabeledData[vid]    = None
        minDistToLabeled[vid] = None

    del (unlabeledData, minDistToLabeled)

    with torch.cuda.device(device):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return videoRank