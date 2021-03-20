import os
import re
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def IMG_SAVER(XL = 336,YL = 192):
    """ This function stores the images in a tensor (INPUT_SAVE). Each line is a different image.
    This function also calls the READER function from TXT_READER, meaning it will also create the
    Output tensor (OUT) associated to each image automatically, as well as the IDX list listing
    the files saved.

    :param XL: Image's width
    :param YL: Image's height
    :return: INPUT_SAVE: A tensor with every image saved. Shape: [Nb of images, XL*YL]
             OUT: The tensor storing all of the outputs (rectangular masks) for each image. Shape: [Nb of images, 2*XL*YL]
             IDX: A list containing the the valid scenes file name, in the order they appear in the tensors.
    """
    from TXT_READER import READER

    OUT,IDX = READER() # Read the files in the folder

    inv_dict = {i: IDX[i] for i in range(len(IDX))} # Dictionary linking the line of the tensor OUT to a file name
    # example: inv_dict[0] = "scene00001.txt"

    INPUT_SAVE = th.zeros(len(OUT),XL*YL) # for saving the images in only one tensor
    # It's useful to save them like this because this way they'll be more easily called for batch training later

    for N in range(len(OUT)):
        filename = inv_dict[N]              # We find the file's name from the line number
        filename = filename[0:-4] + '.png'  # We change '.txt' to '.png'
        try:
            Img = mpimg.imread(filename)    # Try to same the image into Img
        except FileNotFoundError:
            print('File not Found: ',filename)  # In case the file is not in the folder, which is weird
            continue
        else:
            Img = Img[:, :, 0]              # We only need one of the color channels since it's grey scale
            Img = th.from_numpy(Img)        # We turn Img into a th.tensor
            INPUT_SAVE[N,:] = Img.view(XL*YL)   # We save the image in the INPUT_SAVE

    return INPUT_SAVE, OUT, IDX

def IMG_PLOTTER(N,INPUT_SAVE,OUT,IDX,XL = 336, YL = 192):
    """ This function serves to plot a scene with the rectangular filters.

    :param N: Scene number
    :param INPUT_SAVE: The image tensor
    :param OUT: The rectangular filter tensor
    :param IDX: The files read
    :param XL: Image's width
    :param YL: Image's height
    :return: None
    """
    from TXT_READER import FILENAME_TXT
    # For plotting
    test_file = FILENAME_TXT(N)                         # The file name associated to scene N
    IDX_dict = {IDX[i]: i for i in range(len(IDX))}     # Dictionary connecting filename and
    # the line in the OUTPUT_SAVE matrix.
    NB = IDX_dict[test_file]
    OUT_0 = OUT[NB].view(2,XL,YL)    # We separate the image we want to check out
    Img_w_filter = INPUT_SAVE[NB].view(YL,XL) + OUT_0[0].T - OUT_0[1].T  # For showing the rectangles on top of the drops
    # IMPORTANT: NOTICE THAT WE HAVE TO TRANSPOSE OUT_0.
    plt.imshow(Img_w_filter)  # Plotting
    plt.show()  # Showing

    return None

if __name__ == '__main__':
    IN,OUT,IDX = IMG_SAVER()
    #IMG_PLOTTER(1629,IN,OUT,IDX) # Plotting for scene nb 1625

