import numpy as np
import multiprocessing as mp


def tonumpyarray(
    mp_arr,
):  # We are pluggin this function here so that we dont need to import it.
    # mp_array is a shared memory array with lock
    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)


def initialize_pool(shared_array, img, filt):
    """
    This function defines the global variables that will be used in the
    filtering process, that is for each pool the global variables that will be used

    Inputs:
    shared_array: space where the threads will store the results
    img: image to be filtered
    filt: the filter we will apply to the image
    """

    # global variables used
    global shared_space  # reference space where the matrix with the results are stored
    global shared_matrix  # matrix where all the threads will write the filtered value of a pixel
    global image  # the image that to be filtered
    global my_filter  # filter that is applied to the image

    # assignment to the global variables
    image = img
    my_filter = filt
    size = image.shape
    shared_space = shared_array
    shared_matrix = tonumpyarray(shared_space).reshape(size)


def filter_image(row):
    """
    Big function that does the filtering for each row given a specific filter depending of
    it size from 1 to 5 rows or columns. It will used locks to be able to write to the global
    variable in a secure way.

    Inputs:
    row --> row of the image to be filtered (row array of the image)
    """

    # We first call all the global variables to be used
    global my_filter  # The filter to be applied
    global shared_space  # The space where to store the resulting value
    global image  # The image to be filtered

    image = image
    (rows, cols, depth) = image.shape
    (filter_rows, filter_cols) = my_filter.shape

    # In order to adapt the function to be able to use filters from 1 to 5 rows/columns
    # we came to the idea that the best way was to consider the largest possible filter
    # size. Before multipling the filter we adapt each filter to perform the multiplication
    # appropiatly.

    srow = image[row, :, :]  # The current row that is being filtered

    # top row given a 5x5 filter
    prevrow_index = 2 if row >= 2 else 0
    prevprow = image[row - prevrow_index, :, :]

    # second row given a 5x5 filter
    prow_index = 1 if row >= 1 else 0
    prow = image[row - prow_index, :, :]

    # forth row given a 5x5 filter
    nrow_index = 1 if row < rows - 1 else 0
    nrow = image[row + nrow_index, :, :]

    # last row given a 5x5 filter
    nextnrow_index = 2 if row < rows - 2 else 0
    nextnrow = image[row + nextnrow_index, :, :]

    frow = np.zeros_like(srow)  # We initialize the row where we will store the

    # We first iterate through the colors and then thorugh all the pixels of the row
    # Matrix will store the values of the pixels that will be used if the filter was 5x5

    for d in range(depth):
        for c in range(cols):
            if c == 0:  # we are in the left border
                matrix = np.array(
                    (
                        [
                            prevprow[c][d],
                            prevprow[c][d],
                            prevprow[c][d],
                            prevprow[c + 1][d],
                            prevprow[c + 2][d],
                        ],
                        [
                            prow[c][d],
                            prow[c][d],
                            prow[c][d],
                            prow[c + 1][d],
                            prow[c + 2][d],
                        ],
                        [
                            srow[c][d],
                            srow[c][d],
                            srow[c][d],
                            srow[c + 1][d],
                            srow[c + 2][d],
                        ],
                        [
                            nrow[c][d],
                            nrow[c][d],
                            nrow[c][d],
                            nrow[c + 1][d],
                            nrow[c + 2][d],
                        ],
                        [
                            nextnrow[c][d],
                            nextnrow[c][d],
                            nextnrow[c][d],
                            nextnrow[c + 1][d],
                            nextnrow[c + 2][d],
                        ],
                    )
                )

            elif c == 1:  # we are in the second col of the left border
                matrix = np.array(
                    (
                        [
                            prevprow[c - 1][d],
                            prevprow[c - 1][d],
                            prevprow[c][d],
                            prevprow[c + 1][d],
                            prevprow[c + 2][d],
                        ],
                        [
                            prow[c - 1][d],
                            prow[c - 1][d],
                            prow[c][d],
                            prow[c + 1][d],
                            prow[c + 2][d],
                        ],
                        [
                            srow[c - 1][d],
                            srow[c - 1][d],
                            srow[c][d],
                            srow[c + 1][d],
                            srow[c + 2][d],
                        ],
                        [
                            nrow[c - 1][d],
                            nrow[c - 1][d],
                            nrow[c][d],
                            nrow[c + 1][d],
                            nrow[c + 2][d],
                        ],
                        [
                            nextnrow[c - 1][d],
                            nextnrow[c - 1][d],
                            nextnrow[c][d],
                            nextnrow[c + 1][d],
                            nextnrow[c + 2][d],
                        ],
                    )
                )

            elif c == (cols - 2):  # we are in the right border
                matrix = np.array(
                    (
                        [
                            prevprow[c - 2][d],
                            prevprow[c - 1][d],
                            prevprow[c][d],
                            prevprow[c + 1][d],
                            prevprow[c + 1][d],
                        ],
                        [
                            prow[c - 2][d],
                            prow[c - 1][d],
                            prow[c][d],
                            prow[c + 1][d],
                            prow[c + 1][d],
                        ],
                        [
                            srow[c - 2][d],
                            srow[c - 1][d],
                            srow[c][d],
                            srow[c + 1][d],
                            srow[c + 1][d],
                        ],
                        [
                            nrow[c - 2][d],
                            nrow[c - 1][d],
                            nrow[c][d],
                            nrow[c + 1][d],
                            nrow[c + 1][d],
                        ],
                        [
                            nextnrow[c - 2][d],
                            nextnrow[c - 1][d],
                            nextnrow[c][d],
                            nextnrow[c + 1][d],
                            nextnrow[c + 1][d],
                        ],
                    )
                )

            elif c == (cols - 1):  # we are in the second col of the right border
                matrix = np.array(
                    (
                        [
                            prevprow[c - 2][d],
                            prevprow[c - 1][d],
                            prevprow[c][d],
                            prevprow[c][d],
                            prevprow[c][d],
                        ],
                        [
                            prow[c - 2][d],
                            prow[c - 1][d],
                            prow[c][d],
                            prow[c][d],
                            prow[c][d],
                        ],
                        [
                            srow[c - 2][d],
                            srow[c - 1][d],
                            srow[c][d],
                            srow[c][d],
                            srow[c][d],
                        ],
                        [
                            nrow[c - 2][d],
                            nrow[c - 1][d],
                            nrow[c][d],
                            nrow[c][d],
                            nrow[c][d],
                        ],
                        [
                            nextnrow[c - 2][d],
                            nextnrow[c - 1][d],
                            nextnrow[c][d],
                            nextnrow[c][d],
                            nextnrow[c][d],
                        ],
                    )
                )

            else:  # we are not in a border
                matrix = np.array(
                    (
                        [
                            prevprow[c - 2][d],
                            prevprow[c - 1][d],
                            prevprow[c][d],
                            prevprow[c + 1][d],
                            prevprow[c + 2][d],
                        ],
                        [
                            prow[c - 2][d],
                            prow[c - 1][d],
                            prow[c][d],
                            prow[c + 1][d],
                            prow[c + 2][d],
                        ],
                        [
                            srow[c - 2][d],
                            srow[c - 1][d],
                            srow[c][d],
                            srow[c + 1][d],
                            srow[c + 2][d],
                        ],
                        [
                            nrow[c - 2][d],
                            nrow[c - 1][d],
                            nrow[c][d],
                            nrow[c + 1][d],
                            nrow[c + 2][d],
                        ],
                        [
                            nextnrow[c - 2][d],
                            nextnrow[c - 1][d],
                            nextnrow[c][d],
                            nextnrow[c + 2][d],
                            nextnrow[c + 2][d],
                        ],
                    )
                )

            # Now, once we have the largest posible matrix, depending on the filter
            # size, we will ignore some values so that they are not taken
            # into account in the filtering process. This is done because otherwise,
            # the 1 and the 3 filtes will take a lot of time.

            x = 0  # Case when the filter is 5x5
            if filter_rows == 3:
                x = 1  # Case when the filter is 3x...
            elif filter_rows == 1:
                x = 2  # Case when the filter is 1x...

            y = 0  # Case when the filter is 5x5
            if filter_cols == 3:
                y = 1  # Case when the filter is ...x3
            elif filter_cols == 1:
                y = 2  # Case when the filter is ...x1

            accu = 0
            for row_ in range(filter_rows):
                for col_ in range(filter_cols):
                    # Multiply those pixels inside the filter size region
                    accu += matrix[row_ + x][col_ + y] * my_filter[row_][col_]

            frow[c, d] = accu

    with shared_space.get_lock():
        shared_matrix[row, :, :] = frow
        # We store the result in the global variable shared matrix using locks
        # to avoid race conditions as we are using multiple threads that will
        # access that variable at the same time.
    return frow


def image_filter(image, filter_mask, numprocessors, filtered_image):
    #  image: numpy array,
    #  filter_mask: numpy array 2D,
    #  numprocessors: int
    #  filtered_image: multiprocessing.Array

    # In this function we are using Pool so that we map the function that
    # filters the image with the rows to be filtered in a multiprocess way.
    image_rows = range(image.shape[0])  # 1st is the row
    with mp.Pool(
        processes=numprocessors,
        initializer=initialize_pool,
        initargs=[filtered_image, image, filter_mask],
    ) as p:
        p.map(filter_image, image_rows)
        p.close()


def filters_execution(
    image, filter_mask1, filter_mask2, numprocessors, filtered_image1, filtered_image2
):
    # image: numpy array,
    # filter_mask1: numpy array 2D,
    # filter_mask2: numpy array 2D,
    # numprocessors: int
    # filtered_image1: multiprocessing.Array,
    # filtered_image2: multiprocessing.Array

    # half_processors = int(numprocessors/2) # Half of the processors is assigned to each task.

    # As this function must wait for both processes to be finished, we have to be able to open
    # and close each of them. We are always doing this inside the function named image_filter
    # while calling the p.close() and p.join() methods
    import math

    proc = math.floor(numprocessors / 2)

    filter_masks = [filter_mask1, filter_mask2]
    filtered_images = [filtered_image1, filtered_image2]

    for i in range(len(filter_masks)):
        p = mp.Process(
            target=image_filter,
            args=(
                image,
                filter_masks[i],
                proc,
                filtered_images[i],
            ),
        )

        p.start()
        p.join()


##########################################################################################
# ********** Functions to allow us understand how does Pool function work:     ************
##########################################################################################
# They are not used in this work but they were useful to understand how Pools work

import time
from multiprocessing import Pool


def sum_square(number):
    s = 0
    for i in range(number):
        s += i * i
    return s


def sum_square_with_mp(number):
    start_time = time.time()
    p = Pool()  # by default uses max number of cores
    result = p.map(sum_square, number)  # maps the function and the argument

    p.close()
    p.join()

    end_time = time.time() - start_time
    print(
        f"Processing {len(number)} numbers took {end_time} time using multiprocessing. "
    )


def sum_square_without_mp(number):
    start_time = time.time()
    result = []
    for i in number:
        result.append(sum_square(i))

    end_time = time.time() - start_time
    print(
        f"Processing {len(number)} numbers took {end_time} time using serial processing. "
    )
