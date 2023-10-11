import numpy as np
import multiprocessing as mp


def tonumpyarray(
    mp_arr,
):  # We are pluggin this function here so that we dont need to import it.
    # mp_array is a shared memory array with lock
    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)


def initialize_pool(shared_array, srcimg, imgfilter):
    """
    This function defines the global variables that will be used in the
    filtering process.

    INPUTS
    shared_array : space where the threads will store the results
    srcimg --> Array: image to be filtered
    imgfilter --> Array: the filter we will apply to the image

    """

    # All the global variables that will be used in the filtering process are initialized here:
    global shared_space  # --> reference space where the matrix with the results are stored
    global shared_matrix  # --> matrix where all the threads will write the filtered value of a pixel
    global image  # --> the image that to be filtered
    global my_filter  # --> the filter that is applied to the image

    # All the global variables that will be used in the filtering process are defined here:
    image = srcimg
    my_filter = imgfilter
    size = image.shape
    shared_space = shared_array
    shared_matrix = tonumpyarray(shared_space).reshape(size)


def filter_image(row):
    """
    This function is in charge of filtering a row of the original image. It uses
    the filter stored as a global variable, and depending on its size, will
    consider 1,2,3,4 or 5 rows/columns to perform the multiplications.
    The image is also stored as a global variable, so that all the threads
    can access it.

    INPUTS
    row --> integer: row of the image to be filtered

    OUTPUT
    Not a direct output, but after this function, one row of the original image
    is filtered. The result is stored in the global shared_matrix using locks
    so that there are not race conditions.

    """
    # We first call all the global variables to be used  (defined in the Pool_init 1 function)
    global my_filter  # The filter to be applied
    global shared_space  # The space where to store the resulting value
    global image  # The image to be filtered

    image = image
    (rows, cols, depth) = image.shape
    (filter_rows, filter_cols) = my_filter.shape

    # Firstly, in order to develop an algorithm capable of filtering an image no
    # matter of what size the filter has (could be 1, 3, 5 x 1, 3, 5), we came to
    # the conclusion that the best idea was to consider the largest possible
    # filter size, and prior to the matrix multiplication process substitute by
    # 0s those parts that were left out of our filter regarding its size.
    # Therefore, we now compute the filtering area as if the filter was 5x5. To
    # do this, we need to find the two previous and next rows of with respect to
    # the row being filtered.

    srow = image[row, :, :]  # The current row that is being filtered

    if (
        row > 1
    ):  # The row is not close to the upper border -->  We can use the previous previous row
        prevprow = image[row - 2, :, :]
    else:  # The row is close to the upper border -->  We cannot use the previous previous row
        prevprow = image[row, :, :]

    if (
        row > 0
    ):  # The row is not close to the upper border -->  We can use the previous row
        prow = image[row - 1, :, :]
    else:  # The row is not close to the upper border -->  We can use the previous row
        prow = image[row, :, :]

    if row == (
        rows - 1
    ):  # The row is close to the lower border -->  We cannot use the next row
        nrow = image[row, :, :]
    else:  # The row is not close to the lower border -->  We can use the next row
        nrow = image[row + 1, :, :]

    if row >= (
        rows - 2
    ):  # The row is close to the lower border -->  We cannot use the next next row
        nextnrow = nrow
    else:  # The row is not close to the lower border -->  We can use the next next row
        nextnrow = image[row + 2, :, :]

    frow = np.zeros_like(srow)  # We initialize the row where we will store the
    # results of the filtered pixels of the row

    # We first iterate through the dimensions and the thorugh all the pixels of the row
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

            elif c == (cols - 2):  # we are in the left border
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

            elif c == (cols - 1):  # we are in the previous column of the left border
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
            # into account in the filtering process. In order to do that, we
            # start multiplying

            x = 0
            y = 0  # Case when the filter is 5x5
            if filter_rows == 3:
                x = 1  # Case when the filter is 3x...
            if filter_cols == 3:
                y = 1  # Case when the filter is ...x3
            if filter_rows == 1:
                x = 2  # Case when the filter is 1x...
            if filter_cols == 1:
                y = 2  # Case when the filter is ...x1

            accu = 0
            for row_ in range(filter_rows):
                for col_ in range(filter_cols):
                    # We just multiply those pixels inside the filter size region
                    # For instance, if the filter is 3x3, the element matrix[0][0] is not used in the
                    # multiplication, since x and y are 1
                    accu += matrix[row_ + x][col_ + y] * my_filter[row_][col_]

            frow[c, d] = accu

    with shared_space.get_lock():
        shared_matrix[row, :, :] = frow
        # We store the result in the global variable shared matrix
        # To avoid race conditions, we use locks so that no more than one thread
        # can modify the variable at the same time

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
