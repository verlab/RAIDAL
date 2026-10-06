"""
CTC-BADGE for CTC-based CSLR.

This file implements CTC-BADGE from Sec. 4.2.3.
We adapt BADGE (Ash et al. 2020) to CSLR by computing gradient embeddings from
the CTC loss, using the top decoded beam as a pseudo-label, followed by
k-means++ batch selection under a frame budget.

To account for variable video lengths, the loss is
always divided by the video's raw annotation cost before computing the
gradient, like so:

    loss_i = CTC(logits_i, decoded_pseudo_transcript_i) / framesInVideo[i]

This makes the BADGE embedding represent expected model update per raw annotation frame rather 
than per video.

One further, independent extension is available as a flag:

    useRaidalFilter (bool):
        This is the "CTC-BADGE + TROI" ablation from Appendix E. The CTC loss
        (and its gradient) is still computed over the full sequence, but only
        the RAIDAL-filtered timesteps contribute to the gradient embeddings used
        by k-means++.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import numpy as np
import torch.nn.functional as F

from active_learning_modules.raidal import expandTimesteps

Array = np.ndarray


def decodedTopBeamToIds(decodedSeqs, gloss2IndexDict: Dict[str, int]) -> List[int]:
    """
    Convert the top CTC beam hypothesis from gloss strings to class IDs.

    Returns an empty list if the decoded sequence is missing or contains unknown glosses.
    """
    if decodedSeqs is None or len(decodedSeqs) == 0:
        return []

    topBeam = decodedSeqs[0]

    if topBeam is None or len(topBeam) == 0:
        return []

    try:
        return [int(gloss2IndexDict[g]) for g in topBeam]
    except KeyError:
        return []


def ctcDeltaFromLogits(
    logitsNp: Array,
    targetIds: Sequence[int],
    device: str,
    blank: int = 0,
    lossScale: float = 1.0,
    zeroInfinity: bool = True,
) -> Optional[Array]:
    """
    Compute delta = d CTC_loss / d logits for one video.

    Args:
        logitsNp: (T, C) saved classifier logits.
        targetIds: decoded pseudo transcript as class IDs, without blanks.
        device: The torch device used for all tensor operations.
        blank: The CTC blank class index.
        lossScale: Scalar multiplier applied to the CTC loss before taking
            gradients. CTC-BADGE always calls this with 1 / rawNumFrames (see
            module docstring); this function itself stays generic and simply
            applies whatever scale it's given.
        zeroInfinity: Forwarded to torch.nn.CTCLoss; guards against the rare
            case where the input is too short for the target, which would
            otherwise produce an infinite loss (and an infinite/NaN gradient).

    Returns:
        delta as a NumPy array with shape (T, C), or None if the pseudo-target is empty.
    """
    if len(targetIds) == 0:
        return None

    logitsNp = np.asarray(logitsNp, dtype=np.float32)

    if logitsNp.ndim != 2:
        raise ValueError(f"Expected logits with shape (T, C), got {logitsNp.shape}")

    numTimesteps = int(logitsNp.shape[0])

    if numTimesteps <= 0:
        return None

    logits = torch.as_tensor(logitsNp, dtype=torch.float32, device=device).clone().detach()
    logits.requires_grad_(True)

    # PyTorch CTC expects log-probabilities shaped as (T, N, C), where N is batch size.
    # N = 1 because each saved feature file corresponds to one video.
    logProbs = F.log_softmax(logits, dim=-1).unsqueeze(1)  # (T, 1, C)

    # CTC target format for a batch of size 1.
    targetsTensor = torch.as_tensor(targetIds, dtype=torch.long, device=device)
    inputLengths  = torch.tensor([numTimesteps], dtype=torch.long, device=device)
    targetLengths = torch.tensor([len(targetIds)], dtype=torch.long, device=device)

    ctcLossFn = torch.nn.CTCLoss(
        blank=blank,
        reduction="none",
        zero_infinity=zeroInfinity,
    )

    # reduction="none" returns shape (1,), so sum() gives a scalar.
    loss = ctcLossFn(logProbs, targetsTensor, inputLengths, targetLengths).sum()

    # CTC-BADGE always calls this with lossScale = 1 / framesInVideo[videoId]
    # (see module docstring); a plain caller could still pass 1.0 for no scaling.
    loss = loss * float(lossScale)

    # delta is the CTC gradient with respect to every class logit.
    delta = torch.autograd.grad(loss, logits, retain_graph=False, create_graph=False)[0]

    return delta.detach().cpu().numpy().astype(np.float32)


def projectedEffectiveClassifierGradient(
    zNp: Array,
    deltaNp: Array,
    projL: Array,
    projR: Array,
) -> Array:
    """
    Compute a projected version of the effective final-layer gradient.

    Returns:
        flattened projected gradient with shape (k_feat * k_cls,)
    """
    zNp = np.asarray(zNp, dtype=np.float32)
    deltaNp = np.asarray(deltaNp, dtype=np.float32)

    if zNp.ndim != 2:
        raise ValueError(f"Expected features with shape (T, d), got {zNp.shape}")
    if deltaNp.ndim != 2:
        raise ValueError(f"Expected delta with shape (T, C), got {deltaNp.shape}")
    if zNp.shape[0] != deltaNp.shape[0]:
        raise ValueError(
            f"Feature/logit length mismatch: features T={zNp.shape[0]}, delta T={deltaNp.shape[0]}"
        )

    zP = zNp @ projL.T          # (T, k_feat)
    dP = deltaNp @ projR.T      # (T, k_cls)

    projectedG = zP.T @ dP      # (k_feat, k_cls)

    return projectedG.reshape(-1).astype(np.float32)


def kMeansPlusPlus(
    embeddings: Dict[str, Array],
    ids: List[str],
    framesInVideo: Dict[str, int],
    frameBudget: int,
    rng: np.random.Generator,
) -> List[str]:
    """
    BADGE-style k-means++ selection under a frame budget.

    First center: largest gradient norm.
    Later centers: D^2 sampling using squared distance to the closest selected center.
    Stop once selected videos exhaust the frame budget.
    """
    if len(ids) == 0:
        return []

    stackedEmbeddings = np.stack([embeddings[i] for i in ids]).astype(np.float32)  # (numVideos, D)
    numVideos = stackedEmbeddings.shape[0]

    if frameBudget is None:
        frameBudget = sum(int(framesInVideo[i]) for i in ids)

    # BADGE starts from the point with the largest gradient norm.
    firstIdx = int(np.linalg.norm(stackedEmbeddings, axis=1).argmax())
    chosenIdxs = [firstIdx]
    chosenIdxSet = {firstIdx}

    closestSq = np.sum((stackedEmbeddings - stackedEmbeddings[firstIdx]) ** 2, axis=1)
    addedFrames = int(framesInVideo[ids[firstIdx]])

    while addedFrames < frameBudget and len(chosenIdxs) < numVideos:
        total = float(closestSq.sum())

        if total <= 0.0:
            # All remaining points coincide with a selected center.
            remaining = [j for j in range(numVideos) if j not in chosenIdxSet]
            nextIdx = int(rng.choice(remaining))
        else:
            probs = closestSq / total
            nextIdx = int(rng.choice(numVideos, p=probs))

            # Numerical safety: chosen points should have zero probability, but guard anyway.
            if nextIdx in chosenIdxSet:
                remaining = [j for j in range(numVideos) if j not in chosenIdxSet]
                nextIdx = int(rng.choice(remaining))

        chosenIdxs.append(nextIdx)
        chosenIdxSet.add(nextIdx)

        addedFrames += int(framesInVideo[ids[nextIdx]])
        closestSq = np.minimum(closestSq, np.sum((stackedEmbeddings - stackedEmbeddings[nextIdx]) ** 2, axis=1))

    return [ids[j] for j in chosenIdxs]


def ctcBadgeImpl(
    unlabeledSet: dict,
    gloss2IndexDict: Dict[str, int],
    framesInVideo: Dict[str, int],
    frameBudget: int,
    device: str = "cuda",
    projFeat: int = 256,
    projClass: int = 256,
    seed: int = 0,
    blank: int = 0,
    zeroInfinity: bool = True,
    useRaidalFilter: bool = False,
    raidalTemporalExpansionRadius: int = 1,
    raidalNumHypothesesUsed: int = 10
) -> dict:
    """
    Shared implementation for CTC-BADGE.

    See the module docstring for the per-frame loss normalization and for
    what `useRaidalFilter` changes.

    Args:
        unlabeledSet (dict): Maps video id -> dict with at least the keys
            'biLSTMFeaturesNoDropout' (per-frame features, shape (T, d)),
            'classifierFeaturesNoDropout' (per-frame, pre-softmax CTC class
            logits, shape (T, C)), and 'ctcDecodedSeqsNoDropout' (list of
            decoded gloss sequences, one per beam, best-first). If
            `useRaidalFilter` is set, also needs 'ctcTimestepsNoDropout' (list
            of per-beam alignment-peak tensors), for every candidate video.
        gloss2IndexDict (dict): Maps gloss string -> integer class index.
        framesInVideo (dict): Maps video id -> raw frame count, used both to
            scale the CTC loss (see module docstring) and to track
            `frameBudget`.
        frameBudget (int): Total number of raw frames to select before
            `kMeansPlusPlus` stops.
        device (str): The torch device used for all tensor operations.
        projFeat (int): Output dimension of the feature-side random projection
            (k_feat in `projectedEffectiveClassifierGradient`'s docstring).
        projClass (int): Output dimension of the class-side random projection
            (k_cls in `projectedEffectiveClassifierGradient`'s docstring).
        seed (int): Seed for the random projections and for k-means++ sampling.
        blank (int): The CTC blank class index.
        zeroInfinity (bool): Forwarded to `ctcDeltaFromLogits`.
        useRaidalFilter (bool): See module docstring / Appendix E.
        raidalTemporalExpansionRadius (int): Temporal Expansion Radius R
            (Eq. 1), forwarded to `expandTimesteps` when `useRaidalFilter` is
            set.
        raidalNumHypothesesUsed (int): How many of the top decoded beams
            contribute alignment peaks to the TROI when `useRaidalFilter` is
            set (see raidal.raidal's docstring for the same parameter).

    Returns:
        dict: Maps video id -> acquisition score. Selected videos get
            descending scores in k-means++ selection order (most informative
            first); videos left unselected once the frame budget is exhausted
            get 0.0.
    """
    ids = list(unlabeledSet.keys())

    if len(ids) == 0:
        return {}

    rng = np.random.default_rng(seed)

    sampleVideoId = ids[0]
    featureDim = int(np.asarray(unlabeledSet[sampleVideoId]["biLSTMFeaturesNoDropout"]).shape[1])
    numClasses = int(np.asarray(unlabeledSet[sampleVideoId]["classifierFeaturesNoDropout"]).shape[1])

    # Two-sided Gaussian random projection.
    # This avoids materializing the full featureDim*numClasses gradient vector.
    projL = (rng.standard_normal((projFeat, featureDim)) / np.sqrt(projFeat)).astype(np.float32)
    projR = (rng.standard_normal((projClass, numClasses)) / np.sqrt(projClass)).astype(np.float32)

    embeddings = {}
    sizeReductions = []  # tracks how much useRaidalFilter shrinks each video's
                         # gradient embedding, for the diagnostic printout below
    for vid in ids:
        zNp = np.asarray(unlabeledSet[vid]["biLSTMFeaturesNoDropout"], dtype=np.float32)
        logitsNp = np.asarray(unlabeledSet[vid]["classifierFeaturesNoDropout"], dtype=np.float32)

        targetIds = decodedTopBeamToIds(
            unlabeledSet[vid].get("ctcDecodedSeqsNoDropout", None),
            gloss2IndexDict,
        )

        # Normalize by the video's raw annotation cost so the
        # gradient represents expected update per frame, not per video.
        # max(..., 1.0) just guards against a pathological zero-frame entry.
        annotationCost = max(float(framesInVideo[vid]), 1.0)
        lossScale = 1.0 / annotationCost

        deltaNp = ctcDeltaFromLogits(
            logitsNp=logitsNp,
            targetIds=targetIds,
            device=device,
            blank=blank,
            lossScale=lossScale,
            zeroInfinity=zeroInfinity,
        )

        if deltaNp is None:
            embeddings[vid] = np.zeros(projFeat * projClass, dtype=np.float32)
            continue

        if useRaidalFilter:
            timesteps = torch.cat(unlabeledSet[vid]['ctcTimestepsNoDropout'][0 : raidalNumHypothesesUsed]).unique().to(torch.int64)
            timesteps = expandTimesteps(timesteps, len(zNp), raidalTemporalExpansionRadius)

            # The CTC loss/gradient above was computed over the
            # FULL sequence; only now do we restrict which timesteps feed into
            # the gradient embedding.
            sizeReductions.append(len(zNp[timesteps]) / len(zNp))
            zNp     = zNp[timesteps]
            deltaNp = deltaNp[timesteps]

        embeddings[vid] = projectedEffectiveClassifierGradient(
            zNp=zNp,
            deltaNp=deltaNp,
            projL=projL,
            projR=projR,
        )

    selected = kMeansPlusPlus(
        embeddings=embeddings,
        ids=ids,
        framesInVideo=framesInVideo,
        frameBudget=frameBudget,
        rng=rng,
    )

    videoRank = {vid: 0.0 for vid in ids}
    for order, vid in enumerate(selected):
        videoRank[vid] = float(len(selected) - order)

    if useRaidalFilter:
        avgReduction        = np.mean(sizeReductions)*100
        stdReduction        = np.std(sizeReductions)*100
        Q15, Q25, Q75, Q90  = np.quantile(sizeReductions, [0.15, 0.25, 0.75, 0.90]) * 100
        print(f"Avg size after TROI filtering: {avgReduction:.2f}% +- {stdReduction:.2f}% of original video. Q15 = {Q15:.2f}%, Q25 = {Q25:.2f}%, Q75 = {Q75:.2f}%, Q90 = {Q90:.2f}%")


    return videoRank


def ctcBadge(
    unlabeledSet: dict,
    gloss2IndexDict: Dict[str, int],
    framesInVideo: Dict[str, int],
    frameBudget: int,
    device: str = "cuda",
    projFeat: int = 256,
    projClass: int = 256,
    seed: int = 0,
    blank: int = 0,
    useRaidalFilter: bool = False,
    raidalTemporalExpansionRadius: int = 1,
    raidalNumHypothesesUsed: int = 10
) -> dict:
    """
    CTC-BADGE described in Sec. 4.2.3. This is our adaptation of BADGE to CSLR via CTC gradient embeddings.

    Uses the top decoded beam as a pseudo-target for the CTC loss, normalized
    by the video's raw frame count.
        loss_i = CTC(logits_i, decoded_pseudo_transcript_i) / framesInVideo[i]

    BADGE embedding:
        g_i = gradient of loss_i wrt the effective final classifier matrix

    Args:
        See `ctcBadgeImpl`, which this simply forwards to (with
        `zeroInfinity` fixed at its default, True).

    Returns:
        dict: Maps video id -> acquisition score (see `ctcBadgeImpl`).
    """
    return ctcBadgeImpl(
        unlabeledSet=unlabeledSet,
        gloss2IndexDict=gloss2IndexDict,
        framesInVideo=framesInVideo,
        frameBudget=frameBudget,
        device=device,
        projFeat=projFeat,
        projClass=projClass,
        seed=seed,
        blank=blank,
        useRaidalFilter=useRaidalFilter,
        raidalTemporalExpansionRadius=raidalTemporalExpansionRadius,
        raidalNumHypothesesUsed=raidalNumHypothesesUsed
    )
