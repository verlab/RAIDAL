import numpy as np

from active_learning_modules.utils import softmax


def entropy(unlabeledSet: dict) -> dict:
    """
    Calculates sequence-level uncertainty using mean per-timestep entropy.

    The classifier logits are converted to class probabilities independently
    at each timestep. Entropy is then calculated for every timestep and
    averaged across the temporal dimension to obtain a single acquisition
    score for each video.

    Args:
        unlabeledSet (dict): Dictionary where each key is a video ID. Each
            video must contain "classifierFeaturesNoDropout", holding the
            classifier logits for every timestep.

    Returns:
        dict: Dictionary mapping each video ID to its mean entropy acquisition
            score. Higher scores indicate greater uncertainty.
    """

    def entropyFunction(probs):
        EPS = 1e-12

        # Prevent log2(0). Since clipping slightly changes the probability
        # distribution, normalize each timestep again afterwards.
        probs = np.clip(probs, EPS, 1)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        # Calculate the entropy of the class distribution independently for
        # every timestep.
        H = -np.sum(probs * np.log2(probs), axis=1)

        # Collapse the temporal dimension into a single sequence-level score.
        videoScore = H.mean()

        return videoScore
    
    
    # Convert the classifier logits to per-timestep class probabilities.
    # Keep them in a separate dictionary so the input data is not modified.
    classProbs = {
        videoId: softmax(unlabeledSet[videoId]["classifierFeaturesNoDropout"])
        for videoId in unlabeledSet.keys()
    }

    # Rank each video using its mean per-timestep entropy.
    videoRank = {
        videoId: entropyFunction(classProbs[videoId])
        for videoId in classProbs.keys()
    }

    return videoRank