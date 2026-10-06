import torch
import numpy as np

from active_learning_modules.utils import softmax
from active_learning_modules.raidal import expandTimesteps


def raidal_E(unlabeledSet: dict, temporalExpansionRadius: int, numHypothesesUsed: int) -> dict:
    """
    Calculates the RAIDAL-E acquisition score. This is the "RAIDAL-E" ablation from Sec. 5.4 / Table 6(b)

    RAIDAL-E applies RAIDAL's decoder-based temporal filtering to the standard
    mean entropy acquisition function. Instead of averaging entropy across the
    complete feature sequence, entropy is calculated only over timesteps inside
    the Temporal Region of Interest constructed from the CTC alignment peaks.

    Args:
        unlabeledSet (dict): Dictionary where each key is a video ID. Each
            video must contain the classifier logits in
            "classifierFeaturesNoDropout" and the CTC alignment peaks for each
            decoded hypothesis in "ctcTimestepsNoDropout".
        temporalExpansionRadius (int): Number of feature timesteps retained
            before and after each CTC alignment peak.
        numHypothesesUsed (int): How many of the top-K decoded hypotheses
            (best-first) contribute alignment peaks to the TROI. RAIDAL's default
            setting is K = 10, matching the decoder's beam search width.
    Returns:
        dict: Dictionary mapping each video ID to its RAIDAL-E acquisition
            score. Higher scores indicate greater uncertainty inside the
            decoder-supported temporal regions.
    """

    # Convert the classifier logits into per-timestep class probabilities.
    classProbs = {
        k: softmax(v['classifierFeaturesNoDropout'])
        for k, v in unlabeledSet.items()
    }

    # Apply RAIDAL's Redundancy-Aware Filtering before calculating entropy.
    for k, v in unlabeledSet.items():

        # ctcTimestepsNoDropout contains one tensor of alignment peaks for each
        # stored beam-search hypothesis. Merge the peaks from all hypotheses
        # and remove duplicates to obtain the decoder-supported temporal anchors.
        #
        # This assumes the stored hypotheses correspond to the K hypotheses
        # intended for the experiment (K = 10 in the default RAIDAL setup).
        all_beams_timesteps = torch.cat(v['ctcTimestepsNoDropout'][0 : numHypothesesUsed])
        all_beams_timesteps = all_beams_timesteps.unique()

        timesteps = torch.as_tensor(
            all_beams_timesteps.to(torch.int64)
        )

        # Expand each alignment peak by the temporal radius R and merge the
        # resulting windows into the Temporal Region of Interest.
        timesteps = expandTimesteps(
            timesteps,
            len(classProbs[k]),
            temporalExpansionRadius
        )

        # Discard timesteps outside the decoder-supported region. RAIDAL-E
        # computes uncertainty only from the probabilities retained by T_ROI.
        classProbs[k] = classProbs[k][timesteps]

    def entropyFunction(probs):
        EPS = 1e-12

        # Prevent log2(0). Renormalize afterwards because clipping slightly
        # changes the sum of each probability distribution.
        probs = np.clip(probs, EPS, 1)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        # Calculate entropy independently at every retained timestep.
        H = -np.sum(probs * np.log2(probs), axis=1)

        # Average the retained timestep entropies into a single video score.
        videoScore = H.mean()

        return videoScore

    videoRank = {
        videoId: entropyFunction(classProbs[videoId])
        for videoId in classProbs.keys()
    }

    return videoRank