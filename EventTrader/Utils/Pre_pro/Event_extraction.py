import numpy as np


def decompose_array(array):
    """
    Input:
        [0, 1, 0, 1, 1]
    Output:
        [0, 1, 0, 0, 0;
         0, 0, 0, 1, 0;
         0, 0, 0, 0, 1]
    """
    result = np.zeros((0, len(array)), dtype=int)
    for i, value in enumerate(array):
        if value > 0:
            temp = np.zeros(array.shape, dtype=int)
            temp[i] = 1
            result = np.vstack((result, temp.reshape(1, -1)))
    return result


def my_roll(array, roll):
    """
    Implements a linear shift (i.e. no circular shift)
    """
    rolled_array = np.roll(array, roll)
    if roll > 0:
        rolled_array[:roll] = 0
    else:
        rolled_array[roll:] = 0
    return rolled_array


def distance_to_impulse(array, max_roll=100):
    """
    Input:
        [0, 0, 1, 0, 0]
    Output:
        [2, 1, 0, -1, -2]
    """
    idx_impulse = np.argmax(array)
    result = np.zeros(array.shape, dtype=int)
    for roll in range(1, max_roll + 1):
        result += -roll * my_roll(array, -roll) + roll * my_roll(array, roll)

    if idx_impulse + max_roll < len(array):
        result[idx_impulse + max_roll:] = 1000000
    if idx_impulse - max_roll > 0:
        result[:idx_impulse - max_roll] = 1000000
    return result


def closest_to_zero(array):
    return array[np.argmin(np.abs(array))]


def distance_to_event_numpy(array):
    distances_to_each_event = np.apply_along_axis(distance_to_impulse, axis=1, arr=decompose_array(array))
    distances = np.apply_along_axis(closest_to_zero, axis=0, arr=distances_to_each_event)
    return distances