# utility functions for the robot
import numpy as np
from scipy.ndimage import convolve


# ref: https://github.com/constantinpape/vis_tools/blob/master/vis_tools/edges.py#L5
def make_edges3d(segmentation):
    """ Make 3d edge volume from 3d segmentation
    """
    # NOTE we add one here to make sure that we don't have zero in the segmentation
    gz = convolve(segmentation + 1, np.array([-1.0, 0.0, 1.0]).reshape(3, 1, 1))
    gy = convolve(segmentation + 1, np.array([-1.0, 0.0, 1.0]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1.0, 0.0, 1.0]).reshape(1, 1, 3))
    return (gx ** 2 + gy ** 2 + gz ** 2) > 0


"""
# create patches
def tile_image(image_shape, tile_shape):
     cuts the input image into pieces of size 'tile_size'
    and returns a list of indices conatining the starting index (x,y)
    for each patch

    Args:
    image_shape (tuple): shape of input n-dimensional image
    tile_size (int): cutting parameter
    

    assert image_shape[-1] >= tile_size and image_shape[-2] >= tile_size, "image too small for this tile size"

    tiles = []
    (w, h) = image_shape[-2], image_shape[-1]
    for wsi in range(0, w - tile_size + 1, int(tile_size)):
        for hsi in range(0, h - tile_size + 1, int(tile_size)):
            img = ((wsi, hsi), (wsi + tile_size, hsi + tile_size))
            tiles.append(img)

    if h % tile_size != 0:
        for wsi in range(0, w - tile_size + 1, int(tile_size)):
            img = ((wsi, h - tile_size), (wsi + tile_size, h))
            tiles.append(img)

    if w % tile_size != 0:
        for hsi in range(0, h - tile_size + 1, int(tile_size)):
            img = ((w - tile_size, hsi), (w, hsi + tile_size))
            tiles.append(img)

    if w % tile_size != 0 and h % tile_size != 0:
        img = ((w - tile_size, h - tile_size), (w, h))
        tiles.append(img)

    return tiles
"""


def tile_image(arr_shape, block_shape):
    """ chops of blocks of given size from an array 

    Args:
    arr_shape(tuple): size of input array (ndarray)
    block_shape (tuple): size of block to cut into (ndarray)

    Return type: list(tuple(slice()))- a list of tuples, one per block where each tuple has
    n slice objects, one per dimension (n: number of dimensions)
    """

    assert len(arr_shape) == len(block_shape), "block shape not compatible with array shape"
    for i in range(len(arr_shape)):
        assert arr_shape[i] >= block_shape[i], "block shape not compatible with array shape"

    no_of_blocks = 1
    for i in range(len(block_shape)):
        x = int(arr_shape[i] / block_shape[i])
        if arr_shape[i] % block_shape[i]:
            x += 1
        no_of_blocks *= x

    block_list = []
    for i in range(no_of_blocks):
        block_list.append([])

    for n in range(len(block_shape)):
        j = 0
        for i in range(no_of_blocks):

            if j + block_shape[n] > arr_shape[n]:
                block_list[i].append((arr_shape[n] - block_shape[n], arr_shape[n]))
                break

            block_list[i].append((j, j + block_shape[n]))
            j += block_shape[n]

    for i in range(no_of_blocks):
        for j in range(len(block_list[i])):
            block_list[i][j] = slice(block_list[i][j][0], block_list[i][j][1])

        block_list[i] = tuple(block_list[i])

    return block_list
