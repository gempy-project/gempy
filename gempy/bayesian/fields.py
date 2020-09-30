import numpy as np

from scipy.stats import entropy
import warnings


def compute_prob(blocks):
    warnings.warn("This function is Deprecated, please use the probability function instead")
    return probability(blocks)


def probability(blocks: np.ndarray) -> np.ndarray:
    """
    compute the probabilities for the unique values in the array given
    :param blocks: numpy array of different computed model results
    :return:
    """
    blocks = np.round(blocks)
    return np.mean([blocks == _ for _ in np.unique(blocks)], axis=1)


def calculate_ie_masked(prob):
    warnings.warn("This function is Deprecated, please use the information_entropy function instead")
    return information_entropy(prob)


def information_entropy(probabilities: np.ndarray, base=2) -> np.ndarray:
    """
    Calculate the Information Entropy of the provided probabilities array

    Eq 2. from doi.org/10.1016/j.tecto.2011.05.001

    but using scipy implementation for speed and simplicity

    note the paper assumes base two everywhere

    :param base: base for entropy, defaults to 2 as per paper
    :param probabilities: probabilities array, first axis is for each possible layers (M, n cells)
    :return: entropy of the probability array, flattens the first dimension
    """
    return entropy(probabilities, base=base)


def fuzziness(probabilities: np.ndarray) -> float:
    """
    Return the fuzziness of the probability array

    Eq 3. from doi.org/10.1016/j.tecto.2011.05.001
    :param probabilities: probabilities array
    :return: float of fuzziness
    """
    p = probabilities
    fuzz = -np.mean(np.nan_to_num(p * np.log(p) + (1 - p) * np.log(1 - p)))
    return fuzz


def total_model_entropy(ie: np.ndarray) -> float:
    """
    Return the Total Model Information Entropy (the mean of the array)
    Eq 4. from doi.org/10.1016/j.tecto.2011.05.001
    :param ie: information entropy array
    :return:
    """
    return float(np.mean(ie))
