import pandas as pd
import numpy as np


def probs_to_preds(probs, threshold):
    """
    Convert probability scores from a model to binary predictions based on a threshold or a series of thresholds for each class.

    Parameters
    ==========
    probs : pd.DataFrame
        Probabilities of each observation belonging to each class. Shape should be (#  of observations, #  of classes).
    threshold : float or dict
        If float, score between 0 and 1 to use as a threshold for prediction.
        If dict, iterates across all classes and applies that class-specific threshold.

    Returns
    =======
    preds : pd.DataFrame
        Binary predictions for each observation into each class. Shape should be (#  of observations, #  of classes).
    """

    # Make sure a threshold is specified 
    if threshold >= 0:

        # If float, convert all columns based on same threshold 
        if isinstance(threshold, float):

            preds = pd.DataFrame(np.where(probs > threshold, 1, 0), columns=probs.columns)

            return preds

        # If dict, convert columns based on separate thresholds 
        elif isinstance(threshold, dict):

            # Check if all columns exist in both thresholds and probabilities 
            if list(threshold.keys()) == list(probs.columns):

                preds = probs.copy()

                # Iterate through classes and apply class-specific threshold 
                for class_, thresh in threshold.items():
                    preds[class_] = np.where(preds[class_] > thresh, 1, 0)

                return preds

            else:
                print('Not the same classes in threshold dict and probabilities dataframe.')
                return

        else:
            print("Invalid type for threshold. Should be float or dict.")
            return

    else:
        print("Please specify a threshold or a dictionary of thresholds.")
        return
