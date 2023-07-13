import numpy as np


def convert_label_to_vect(label: int) -> np.ndarray:
    """
    Note that 0 is labelled as 10. 
    We will make the 0th index correspond to zero.
    ***this function has been checked manually, lgtm***
    """
    vect = np.zeros((1,10))
    vect[0][label%10] = 1
    return vect.T # we want a column vector