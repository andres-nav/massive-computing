# /usr/bin/env python

import numpy as np
import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile

F_IMAGE1 = ""  # <DEFINE HERE WHICH IMAGE YOU WANTS TO LOAD
F_IMAGE2 = ""  # <DEFINE HERE WHICH IMAGE YOU WANTS TO LOAD

image1 = np.array(Image.open(F_IMAGE1))
image2 = np.array(Image.open(F_IMAGE1))

filter1 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
filter2 = np.array([0.5, 0, -0.5])
filter3 = np.array([[0.5], [0], [0.5]])

filter4 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
filter5 = np.array(
    [
        [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
        [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
        [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
        [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
        [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
    ]
)

NUMPROCESS = 4

filtered_image1_VECTOR=#HERE YOU HAVE TO DEFINE THE MULTIPROCESSING VECTOR FOR IMAGE1
filtered_image2_VECTOR=#HERE YOU HAVE TO DEFINE THE MULTIPROCESSING VECTOR FOR IMAGE2

def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

# AFTER THIS CELL YOU HAVE TO WRITE YOUR CODE

# After this cell you have to use the preloaded image defined in the F_IMAGE variable, select two of the 5 filter predefined, and using the NUMPROCESS processors, apply the filters to the image and check results.

# * The first filter is impulse response filter (the image output must be equals to the original one).
# * The second filter is an edge filter, first order in x axis,  
# * The third filter is an edge filter, first order in y axis,
# * the fourth filter is an edge filter, second order, bi-directional
# * the fifth filter is a blur gausian filter.

filtered_image1=tonumpyarray(filtered_image1_VECTOR).reshape(image1.shape)
filtered_image2=tonumpyarray(filtered_image2_VECTOR).reshape(image2.shape)

plt.figure()
plt.imshow(filtered_image1)

plt.figure()
plt.imshow(filtered_image2)

