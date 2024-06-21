"""Automatic segment definition subroutines."""

__all__ = ["find_segment_definitions", "find_segments", "split_rotated_segments"]

import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def _find_repeating_pattern(arr):
    n = len(arr)
    for length in range(1, n // 2 + 1):
        is_pattern = True
        for i in range(length, n):
            if arr[i] != arr[i % length]:
                is_pattern = False
                break
        if is_pattern:
            return arr[:length], length
    return None, 0


def find_segment_definitions(arr):
    patterns = []
    start = 0
    while start < len(arr):
        pattern, length = _find_repeating_pattern(arr[start:])
        if pattern is None or length == 0:
            # No more patterns found
            break
        patterns.append(pattern)
        start += length * (len(arr[start:]) // length)
    return patterns


@nb.njit(cache=True, fastmath=True)
def find_segments(array, subarray):
    n = len(array)
    m = len(subarray)
    result = np.zeros(n, dtype=np.bool_)

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if array[i + j] != subarray[j]:
                match = False
                break
        if match:
            for k in range(m):
                result[i + k] = True

    return result


def split_rotated_segments(input):
    output = []
    for segment in input:
        tmp = _split_signed_blocks(segment)
        for item in tmp:
            output.append(item)
    return output


# Here we make the (I think) reasonable assumption that rotated blocks with different angles 
# (e.g., two consecutive (spiral arm, spiral rewinder) pairs with different angles)
# are separated by non rotated blocks (e.g., RF pulses or z-spoilers)
def _split_signed_blocks(arr):
    if not arr:
        return []
    
    # Convert the input list to a numpy array
    arr = np.array(arr)
    
    # Create a sign array where positive numbers are marked as 1, negative as -1
    sign = np.sign(arr)
    
    # Find the indices where the sign changes
    change_indices = np.where(np.diff(sign) != 0)[0] + 1
    
    # Split the array at the change indices
    split_blocks = np.split(arr, change_indices)
    
    # Convert the numpy arrays in the list to python lists
    result = [list(abs(block)) for block in split_blocks]
    
    return result
