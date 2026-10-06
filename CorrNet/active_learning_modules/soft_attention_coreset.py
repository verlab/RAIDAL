import torch
import numpy as np

from tqdm import tqdm

from active_learning_modules.utils import normalizeEmbeddings


def softAttentionCoreset(unlabeledSet: dict, labeledSet: dict, device: str, framesInVideo: dict, frameBudget=None) -> dict:
    """Computes RAIDAL's Information Density score with soft, per-frame CTC weighting.

    This is the "Soft-Attention Core-set" baseline from Sec. 4.2.1: RAIDAL's
    closest point of comparison. Instead of using the CTC decoder as a hard filter
    (RAIDAL's T ROI), every frame of every video is kept, and each frame's
    distance contribution is softly weighted by its non-blank probability (one
    minus the CTC blank probability) rather than being included or excluded
    outright.

    Args:
        unlabeledSet (dict): Maps video id -> dict with at least the keys
            'biLSTMFeaturesNoDropout' (per-frame features, shape (T, D)) and
            'classifierFeaturesNoDropout' (per-frame, pre-softmax CTC class
            logits), for every candidate video.
        labeledSet (dict): Same structure as `unlabeledSet`, but for videos
            already in the labeled pool. Used in full as the reference set F_L.
        device (str): The torch device used for all tensor operations.
        framesInVideo (dict): Maps video id -> raw frame count T_i, used to
            normalize the acquisition score by annotation cost.
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
    
    # Turns each frame's raw CTC classifier output into a soft "is this frame
    # part of a sign?" weight, computed as one minus the frame's blank probability.
    signProbabilities = {}
    for k, v in unlabeledSet.items():
        BLANK_IDX  = 0
        # They are NOT softmaxxed by default. We softmax them here.
        classifierProbs = torch.as_tensor(v['classifierFeaturesNoDropout']).to(device).softmax(axis=-1)
        
        # Get the probabilities of each frame being blank
        blankProbs      = classifierProbs[ : ,  BLANK_IDX]
        # Same thing, but the probability of each frame being a sign
        nonBlankProbs   = 1.0 - blankProbs

        signProbabilities[k] = nonBlankProbs

    scoresBiLSTM = coreset(unlabeledEmbeddingsBiLSTM, labeledEmbeddingsBiLSTM, signProbabilities, device, framesInVideo, frameBudget)

    return scoresBiLSTM


def coreset(unlabeledEmbeddingsBiLSTM: dict, labeledEmbeddingsBiLSTM: dict, signProbabilities: dict, device: str, framesInVideo: dict, frameBudget=None) -> dict:
    """Greedily ranks unlabeled videos by soft-weighted Information Density.

    Same greedy farthest-point procedure as `raidal.coreset` (Sec. 3.2.2, Eq. 4),
    but with no T ROI filtering: every frame contributes to a video's score,
    weighted by `signProbabilities` instead of being kept or dropped outright.

    Args:
        unlabeledEmbeddingsBiLSTM (dict): Maps video id -> L2-normalized
            per-frame embeddings (shape (T, D)) for every candidate video.
        labeledEmbeddingsBiLSTM (dict): Maps video id -> L2-normalized per-frame
            embeddings for every video already in the labeled pool. Used in full
            as the reference set F_L. Re-bound partway through this function to
            the flattened (N, D) tensor built from these per-video embeddings.
        signProbabilities (dict): Maps video id -> per-frame non-blank
            probability (shape (T,)), used to softly weight each frame's
            distance contribution to its video's score.
        device (str): The torch device used for all tensor operations.
        framesInVideo (dict): Maps video id -> raw frame count T_i, used both to
            normalize the acquisition score and to track annotation cost against
            `frameBudget`.
        frameBudget (int, optional): Total number of raw frames to select before
            stopping.

    Returns:
        dict: Maps video id -> acquisition score, in the order videos were
            selected (most informative first).
    """
    
    for videoId, embeddings in tqdm(unlabeledEmbeddingsBiLSTM.items(), desc="Loading videos..."):
        unlabeledEmbeddingsBiLSTM[videoId] = torch.as_tensor(embeddings).float().to(device)
        

    labeledList             = [f for v in labeledEmbeddingsBiLSTM.values() for f in v]
    labeledEmbeddingsBiLSTM = torch.as_tensor(np.array(labeledList)).float().to(device)

    # videoRank is where we store the score of each video. Since Python dicts
    # preserve insertion order, iterating videoRank later also recovers the
    # acquisition order (most informative video first); the score itself is kept
    # mainly for logging/inspection.
    videoRank        = {}
    availableVideos  = set(unlabeledEmbeddingsBiLSTM.keys())

    minDistToLabeled = {}
    
    # Calculate the minimum distance from every unlabeled sample to the labeled
    # set. minDistToLabeled[videoId] holds, per frame of that video, the distance
    # to its nearest neighbor currently in the labeled pool, before any
    # sign-probability weighting is applied.
    for videoId in availableVideos:
        frameFeatures             = unlabeledEmbeddingsBiLSTM[videoId]
        dists                     = torch.cdist(frameFeatures, labeledEmbeddingsBiLSTM)
        minDist, _                = dists.min(dim=1)
        minDistToLabeled[videoId] = minDist

    # We don't need the labeled pool anymore. We can safely delete it from memory.
    del labeledEmbeddingsBiLSTM
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    addedFrames = 0

    # Calculates the score of each video to find the best one
    for _ in tqdm(range(len(unlabeledEmbeddingsBiLSTM)), desc="Ranking videos", unit="video"):
        bestVideoId    = None
        bestVideoScore = -1.0

        for videoId in availableVideos:
            dists          = minDistToLabeled[videoId]
            signProb       = signProbabilities[videoId]
            
            # The soft counterpart to RAIDAL's hard T ROI filter: instead of
            # keeping only decoder-aligned frames, every frame's distance is
            # scaled by how "non-blank" the model thinks it is, so rest poses and
            # pauses still contribute to the score, just less than likely sign
            # frames do.
            weighedDists = dists * signProb

            totalDiversity = torch.sum(weighedDists).item()
            numFrames      = framesInVideo[videoId]

            # For each video, sums its sign-probability-weighted per-frame
            # minimum distances and normalizes by its raw frame count to get its
            # score; the video with the highest score is selected.
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
        # previous minimum is still valid wherever it was already smaller. Note
        # this update runs on the raw (unweighted) distances: sign-probability
        # weighting is re-applied fresh from `signProbabilities` every time a
        # score is computed above, not baked into minDistToLabeled itself.
        newLabeledVideoEmbeddings = unlabeledEmbeddingsBiLSTM[bestVideoId]
        for videoId in availableVideos:
            videoEmbeddings = unlabeledEmbeddingsBiLSTM[videoId]

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
    for vid in unlabeledEmbeddingsBiLSTM:
        unlabeledEmbeddingsBiLSTM[vid]    = None
        minDistToLabeled[vid] = None

    del (unlabeledEmbeddingsBiLSTM, minDistToLabeled)

    with torch.cuda.device(device):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return videoRank