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


# create patches
def tile_image(image_shape, tile_size):
    """ cuts the input image into pieces of size 'tile_size'
    and returns a list of indices conatining the starting index (x,y)
    for each patch

    Args:
    image_shape (tuple): shape of input n-dimensional image
    tile_size (int): cutting parameter
    """

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
    """
    x = []
    for i in range(len(image_shape) - 2):
        x.append([0, image_shape[i]])

    for i in range(len(tiles)):
        tiles[i] = x + tiles[i]
    """
    return tiles
