import torch
import warnings
import torch.nn.functional as F

def uncast(unlabeledSet: dict, gloss2IndexDict: dict, device: str) -> dict:
    """Computes the UNCAST pseudo-label uncertainty acquisition score.

    Implementation of UNCAST from Sec. 4.2.2. This is our adaptation of the
    sequence-level CTC uncertainty estimator from Kim & Lee, "Uncertainty-Aware
    Self-Training for CTC-Based Automatic Speech Recognition" (AAAI 2025).

    Args:
        unlabeledSet (dict): Maps video id -> dict with at least the keys:
            'ctcDecodedSeqsNoDropout' (list of decoded gloss sequences, one per
            beam, ordered best-first -- only the top beam, index 0, is used),
            'classifierFeaturesNoDropout' (per-frame, pre-softmax CTC class
            logits from the no-dropout forward pass, shape (T, numClasses)),
            and 'classifierFeaturesWithDropout' (list of N per-frame logit
            arrays, one per Monte Carlo Dropout forward pass).
        gloss2IndexDict (dict): Maps gloss string -> integer class index,
            needed to turn the decoded gloss sequence into the integer target
            tensor torch.nn.CTCLoss expects.
        device (str): The torch device used for all tensor operations.

    Returns:
        dict: Maps video id -> acquisition score (u_pl, in nats). Videos whose
            top beam decoded to an empty sequence, or to a gloss missing from
            `gloss2IndexDict`, get a score of 0.0 (the minimum possible score
            here), since sequence-level NLL isn't defined for an empty target.
    """

    ctcLossFn = torch.nn.CTCLoss(
        blank=0,
        reduction="none",
        zero_infinity=True
    )

    # videoRank is where we store the score of each video.
    videoRank = {}

    for videoId in unlabeledSet.keys():
        # Get the no-dropout decoded sequence. This is y_hat(x, f),
        # the pseudo-label whose probability we'll score under both the clean
        # and the dropout logits below. Index 0 is the top beam.
        if len(unlabeledSet[videoId]["ctcDecodedSeqsNoDropout"]) == 0:
            videoRank[videoId] = 0.0
            continue

        cleanGlosses = unlabeledSet[videoId]["ctcDecodedSeqsNoDropout"][0]

        # Convert the decoded glosses to class indices, since CTCLoss
        # needs an integer tensor, and not gloss strings.
        try:
            cleanGlossIndices = [gloss2IndexDict[g] for g in cleanGlosses]
        except KeyError:
            # If a decoded gloss isn't in the vocabulary dict (shouldn't happen if
            # the dict is complete), we raise a warning and fail safe rather than crash.
            videoRank[videoId] = 0.0
            warnings.warn("Found a gloss that is not in the vocabulary dict! Please fix me in active_learning_modules.uncast!")
            continue

        if len(cleanGlossIndices) == 0:
            videoRank[videoId] = 0.0
            continue

        # Data uncertainty u_d (Eq. 6). 
        # Score the clean decoded sequence against the no-dropout logits.
        cleanLogits     = unlabeledSet[videoId]["classifierFeaturesNoDropout"]
        dataUncertainty = calculateNormalizedNLL(cleanLogits, cleanGlossIndices, ctcLossFn, device)

        # Model uncertainty u_m (Eq. 7-8): score the SAME clean decoded
        # sequence, but now against each of the N Monte Carlo Dropout logit
        # samples.
        dropoutLogitsSamples = unlabeledSet[videoId]["classifierFeaturesWithDropout"]

        dropoutScores = []
        for noisyLogits in dropoutLogitsSamples:
            score = calculateNormalizedNLL(noisyLogits, cleanGlossIndices, ctcLossFn, device)
            dropoutScores.append(score)

        if len(dropoutScores) > 0:
            modelUncertainty = max(dropoutScores)  # worst-case NLL across the N samples (Eq. 8)
        else:
            modelUncertainty = 0.0

        # Pseudo-label uncertainty u_pl = u_d + u_m (Eq. 9).
        videoRank[videoId] = dataUncertainty + modelUncertainty

    return videoRank

def calculateNormalizedNLL(logits, targetIndices, ctcLossFn, device):
    """Computes -(1/|y|) * log P_CTC(y|logits): length-normalized CTC NLL.

    This is the shared scoring primitive behind both u_d (Eq. 6) and u_m
    (Eq. 7): the two differ only in which logits get passed in here (clean vs.
    a Monte Carlo Dropout sample), never in the target sequence.

    Args:
        logits: Per-frame, pre-softmax CTC class logits, shape (T, numClasses).
        targetIndices (list[int]): The target gloss sequence as class indices.
        ctcLossFn (torch.nn.CTCLoss): A CTCLoss instance configured with
            reduction="none", so it returns one NLL value per call instead of
            averaging over a batch.
        device (str): The torch device used for all tensor operations.

    Returns:
        float: The CTC negative log-likelihood of `targetIndices` given
            `logits`, normalized by the target length |y| so shorter and
            longer decoded sequences stay comparable.
    """
    # CTCLoss expects log-probabilities of shape (T, N, C), so we add the batch
    # dimension (N=1) after log-softmaxing over the class dimension.
    logitsTensor = torch.as_tensor(logits, device=device).float()
    logProbs     = F.log_softmax(logitsTensor, dim=-1).unsqueeze(1) 

    numTimesteps = logProbs.size(0)
    
    # Targets: list[int] -> tensor. With batch size 1, a flat 1D tensor plus
    # matching length tensors is all CTCLoss needs.
    targetsTensor = torch.tensor(targetIndices, device=device).long()
    targetLengths = torch.tensor([len(targetIndices)], device=device).long()
    inputLengths  = torch.tensor([numTimesteps], device=device).long()

    # -log P_CTC(y|logits) (Eq. 2). zero_infinity=True on ctcLossFn guards
    # against the rare case where the input is too short for the target,
    # which would otherwise produce an infinite loss.
    nll = ctcLossFn(logProbs, targetsTensor, inputLengths, targetLengths)
    
    # Make the score length-normalized (Eq. 6-7).
    normalizedNLL = nll.item() / len(targetIndices)
    
    return normalizedNLL