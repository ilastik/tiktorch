# utility functions for the robot
import numpy as np
from scipy.ndimage import convolve
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix


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

def tile_image(arr_shape, block_shape):
     chops of blocks of given size from an array 

    Args:
    arr_shape(tuple): size of input array (ndarray)
    block_shape (tuple): size of block to cut into (ndarray)

    Return type: list(tuple(slice()))- a list of tuples, one per block where each tuple has
    n slice objects, one per dimension (n: number of dimensions)
    

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
"""

n = 0
block_list, idx_list, visited = [], [], {}


def recursive_chop(dim_number, arr_shape, block_shape):
    global block_list, idx_list, visited

    if dim_number >= n:
        return

    for i in range(0, arr_shape[dim_number], block_shape[dim_number]):
        idx_list[dim_number] = i
        recursive_chop(dim_number + 1, arr_shape, block_shape)
        slice_list, visited_key = [], []

        for j in range(n):
            visited_key.append(idx_list[j])
            if idx_list[j] + block_shape[j] > arr_shape[j]:
                slice_list.append(slice(arr_shape[j] - block_shape[j], arr_shape[j]))
            else:
                slice_list.append(slice(idx_list[j], idx_list[j] + block_shape[j]))

        visited_key = tuple(visited_key)
        if visited.get(visited_key) == None:
            visited[visited_key] = 1
            block_list.append(tuple(slice_list))
            
    idx_list[dim_number] = 0


def tile_image(arr_shape, block_shape):
    """
    chops of blocks of given size from an array 

    Args:
    arr_shape(tuple): size of input array (ndarray)
    block_shape (tuple): size of block to cut into (ndarray)

    Return type: list(tuple(slice()))- a list of tuples, one per block where each tuple has
    n slice objects, one per dimension (n: number of dimensions)
    """

    assert len(arr_shape) == len(block_shape), "block shape not compatible with array shape"
    for i in range(len(arr_shape)):
        assert arr_shape[i] >= block_shape[i], "block shape not compatible with array shape"

    global n, idx_list, visited
    n = len(arr_shape)
    block_list.clear(), visited.clear()
    idx_list = [0 for i in range(n)]

    recursive_chop(0, arr_shape, block_shape)
    return block_list


def get_confusion_matrix(pred_labels, act_labels, cls_dict):

    figure_size = (len(cls_dict) * 2, len(cls_dict) * 2)

    act_labels_f = [str(i) for i in np.matrix.flatten(act_labels).tolist()]
    pred_labels_f = [str(i) for i in np.matrix.flatten(pred_labels).tolist()]

    c_mat_arr = confusion_matrix(act_labels_f, pred_labels_f, labels=[str(i) for i in cls_dict.keys()])
    c_mat_p = c_mat_arr / len(act_labels_f)
    c_mat_n = c_mat_arr / np.expand_dims(np.sum(c_mat_arr, axis=1), axis=1)

    # pd_cm = pd.DataFrame(c_mat_p, index=[str(i) for i in cls_dict.values()], columns = [str(i) for i in cls_dict.values()])
    # plt.figure(figsize=figure_size)
    # sn_plot = sn.heatmap(pd_cm, annot=True)
    # sn_plot.figure.savefig(conf_mat_filename)
    #
    pd_cm_n = pd.DataFrame(
        c_mat_n, index=[str(i) for i in cls_dict.values()], columns=[str(i) for i in cls_dict.values()]
    )
    plt.figure(figsize=figure_size)
    sn_plot_n = sn.heatmap(pd_cm_n, annot=True)
    return sn_plot_n.figure
