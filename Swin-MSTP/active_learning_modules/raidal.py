import torch
import numpy as np

from tqdm import tqdm

from active_learning_modules.utils import normalizeEmbeddings


def raidal(unlabeledSet: dict, labeledSet: dict, device: str, framesInVideo: dict, temporalExpansionRadius, numHypothesesUsed, frameBudget=None) -> dict:
    """Runs the RAIDAL acquisition pipeline over a pool of unlabeled videos.

    Prepares the inputs both stages of RAIDAL need -- normalized bi-LSTM features
    and the CTC decoder's alignment peaks -- and delegates the Redundancy-Aware
    Filtering (building each video's TROI) and Information Density scoring
    (Eq. 4, greedy Core-set selection) to `coreset`.

    Args:
        unlabeledSet (dict): Maps video id -> dict with at least the keys
            'biLSTMFeaturesNoDropout' (per-frame features, shape (T, D)) and
            'ctcTimestepsNoDropout' (list of per-beam alignment-peak tensors,
            ordered from the best to the worst beam), for every candidate video.
        labeledSet (dict): Same structure as `unlabeledSet`, but for videos already
            in the labeled pool. Its alignment peaks are never read: the labeled
            pool is used in full, unfiltered, as the reference set F_L in Eq. 4.
        device (str): The torch device used for all tensor operations.
        framesInVideo (dict): Maps video id -> raw frame count T_i, used to
            normalize the acquisition score by annotation cost (Eq. 4).
        temporalExpansionRadius: The Temporal Expansion Radius R (Eq. 1), used to
            grow each alignment peak into a window before building the TROI. This is the value that
            the 'Sensitivity to the Temporal Expansion Radius' ablation changes.
        numHypothesesUsed (int): How many of the top-K decoded hypotheses
            (best-first) contribute alignment peaks to the TROI. RAIDAL's default
            setting is K = 10, matching the decoder's beam search width. This is the value that the
            'Sensitivity to the Number of Decoded Hypotheses' ablation changes.
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

    # Get only the timesteps from the top numHypothesesUsed beams (the decoded
    # hypotheses H from Sec. 3.2.1). The 'num of Beams used' ablation changes 
    unlabeledTimesteps = {}
    for k, v in unlabeledSet.items():
        # v['ctcTimestepsNoDropout'] is a list of tensors (one per beam), each
        # holding that beam's alignment peaks P_h. We concatenate the peaks from
        # the top-K beams and deduplicate with .unique(): this merges P_h across
        # hypotheses (the outer union of Eq. 3 -- the windows themselves are only
        # expanded and merged later, in expandTimesteps).
        all_beams_timesteps   = torch.cat(v['ctcTimestepsNoDropout'][0 : numHypothesesUsed])
        unlabeledTimesteps[k] = all_beams_timesteps.unique()
    
    scoresBiLSTM = coreset(unlabeledEmbeddingsBiLSTM, labeledEmbeddingsBiLSTM, unlabeledTimesteps, device, framesInVideo, temporalExpansionRadius, frameBudget)

    return scoresBiLSTM


def expandTimesteps(timesteps, max_len, temporalExpansionRadius):
    """Builds a video's Temporal Region of Interest (TROI) from its alignment peaks.

    Implements Eq. 2 and the inner union of Eq. 3: each alignment peak is grown
    into a symmetric window [peak - R, peak + R], clipped to the sequence's valid
    range, and the windows from every peak are merged into a single sorted set of
    timestep indices.

    Args:
        timesteps (torch.Tensor or list/array-like of int): The video's alignment
            peaks, already merged across the top-K decoded hypotheses.
        max_len (int): Number of frames in the video's feature sequence, used to
            clip windows so they don't run past its start/end.
        temporalExpansionRadius: The Temporal Expansion Radius R (Eq. 1).

    Returns:
        torch.Tensor: Sorted, deduplicated int64 tensor of timestep indices making
            up the video's TROI.
    """
    if isinstance(timesteps, torch.Tensor):
        timesteps = timesteps.cpu().numpy().tolist()
    
    expanded_set = set()

    for t in timesteps:
        # Clip the window so it stays within the valid frame range.
        start = max(0, int(t) - temporalExpansionRadius)
        end   = min(max_len - 1, int(t) + temporalExpansionRadius)
        
        # range() excludes its upper bound, hence end + 1. Adding to a set both
        # merges overlapping windows from different peaks and drops duplicate
        # indices for free.
        for i in range(start, end + 1):
            expanded_set.add(i)

    timesteps = torch.tensor(sorted(list(expanded_set)), dtype=torch.long)

    return timesteps


def coreset(unlabeledSet: dict, labeledSet: dict, unlabeledTimesteps: dict, device: str, framesInVideo: dict, temporalExpansionRadius, frameBudget=None) -> dict:
    """Filters and greedily ranks unlabeled videos with RAIDAL's Information Density.

    For each unlabeled video, restricts its features to the Temporal Region of
    Interest (TROI) built from its alignment peaks (Redundancy-Aware Filtering,
    Sec. 3.2.1). Then runs a Core-set-style greedy farthest-point selection over
    the TROI-filtered features (Information Density, Sec. 3.2.2, Eq. 4): at each
    step, the video whose filtered features are, on average, farthest from the
    current labeled pool is selected and treated as newly labeled, and the
    remaining videos' scores are updated accordingly. This repeats until
    `frameBudget` raw frames have been selected.

    Args:
        unlabeledSet (dict): Maps video id -> L2-normalized per-frame embeddings
            (shape (T, D)) for every candidate video.
        labeledSet (dict): Maps video id -> L2-normalized per-frame embeddings for
            every video already in the labeled pool. Used in full, unfiltered, as
            the reference set F_L in Eq. 4.
        unlabeledTimesteps (dict): Maps video id -> that video's alignment peak indices
            (merged across the top-K decoded hypotheses), before temporal
            expansion.
        device (str): The torch device used for all tensor operations.
        framesInVideo (dict): Maps video id -> raw frame count T_i, used both to
            normalize the acquisition score (Eq. 4) and to track annotation cost
            against `frameBudget`.
        temporalExpansionRadius: The Temporal Expansion Radius R (Eq. 1), passed
            through to `expandTimesteps` when building each video's TROI.
        frameBudget (int, optional): Total number of raw frames to select before
            stopping.

    Returns:
        dict: Maps video id -> acquisition score (Eq. 4), in the order videos were
            selected (most informative first).
    """

    unlabeledData = {}
    sizeReductions = []
    for videoId, embeddings in tqdm(unlabeledSet.items(), desc="Building TROIs..."):
        embeddings             = torch.as_tensor(embeddings).float().to(device)
        timesteps              = torch.as_tensor(unlabeledTimesteps[videoId].to(torch.int64))
        
        # Grow the raw alignment peaks into the video's TROI (Eq. 2-3).
        timesteps = expandTimesteps(timesteps, len(embeddings), temporalExpansionRadius)

        # Slice the dense per-frame features down to the TROI: this is F-hat from
        # Sec. 3.2.1, the filtered feature sequence Information Density scores are
        # computed over.
        filteredFeatures       = embeddings[timesteps]
        unlabeledData[videoId] = filteredFeatures
        sizeReductions.append(len(unlabeledData[videoId]) / len(embeddings))

    
    # Purely diagnostic: reports how much the TROI filter shrinks each video on
    # average, as a sanity check that filtering is doing meaningful redundancy
    # removal.
    avgReduction        = np.mean(sizeReductions)*100
    stdReduction        = np.std(sizeReductions)*100
    Q15, Q25, Q75, Q90  = np.quantile(sizeReductions, [0.15, 0.25, 0.75, 0.90]) * 100
    
    print(f"Avg size after TROI filtering: {avgReduction:.2f}% +- {stdReduction:.2f}% of original video. Q15 = {Q15:.2f}%, Q25 = {Q25:.2f}%, Q75 = {Q75:.2f}%, Q90 = {Q90:.2f}%")
    # Flatten every labeled video's per-frame features into a single (N, D) pool:
    # this is F_L in Eq. 4. Unlike the unlabeled videos, labeled features are kept
    # in full (no TROI filtering), since they're ground truth rather than
    # candidates being scored.
    labeledList         = [f for v in labeledSet.values() for f in v]
    labeledPoolFeatures = torch.as_tensor(np.array(labeledList)).float().to(device)

    # The videoRank is a dict where we store the score of each video. Since
    # Python dicts preserve insertion order, iterating videoRank later also
    # recovers the acquisition order (most informative video first); the score
    # itself is kept mainly for logging/inspection.
    videoRank = {}
    availableVideos  = set(unlabeledData.keys())

    minDistToLabeled = {}
    
    # Calculate the minimum distance from every unlabeled sample to the labeled
    # set. minDistToLabeled[videoId] holds, per TROI-filtered frame of that
    # video, the distance to its nearest neighbor currently in the labeled pool --
    # the inner min_{f_l in F_L} term of Eq. 4, computed once up front for every
    # video.
    for videoId in availableVideos:
        filteredFeatures          = unlabeledData[videoId]
        dists                     = torch.cdist(filteredFeatures, labeledPoolFeatures)
        minDist, _                = dists.min(dim=1)
        minDistToLabeled[videoId] = minDist

    # We don't need the labeled pool anymore. We can safely delete it from memory.
    del labeledPoolFeatures
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    addedFrames = 0

    # For each video, averages the minDistance of its frames and selects the video with the highest average minDistances matrix.
    for _ in tqdm(range(len(unlabeledData)), desc="Ranking videos", unit="video"):
        bestVideoId    = None
        bestVideoScore = -1.0

        # Calculates the score of each video to find the best one
        for videoId in availableVideos:
            dists = minDistToLabeled[videoId]
            totalDiversity = torch.sum(dists).item()
            numFrames      = framesInVideo[videoId]

            # Eq. 4: sum of nearest-labeled-neighbor distances over the
            # TROI-filtered features, divided by the video's RAW frame count --
            # not the size of the TROI. This is intentional: normalizing by the
            # filtered length instead would let long videos with only a few
            # distant filtered timesteps score artificially high, even though
            # most of their annotation cost comes from frames the decoder never
            # selected as gloss evidence.
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