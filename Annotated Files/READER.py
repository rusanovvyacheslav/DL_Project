import os
import re
import torch as th
import numpy as np
import matplotlib.pyplot as plt


def FILENAME_TXT(N):
    """Constructs the filename according to the scene's number.
    This is necessary since the number of zeros before its number
    will depend on it.

    :param N: the scene number (int) [N < 1000]
    :return: filename: The .txt filename (string)
    """

    if N >= 1000:
        filename = 'scene0' + str(N) + '.txt'  # Filename
    elif N >= 100:
        filename = 'scene00' + str(N) + '.txt'  # Filename
    elif N >= 10:
        filename = 'scene000' + str(N) + '.txt'  # Filename
    else:
        filename = 'scene0000' + str(N) + '.txt'  # Filename

    return filename


def READER(N_Img=4859, XL=336, YL=192):
    """ This function serves to interpret .txt files for the project
    and to save all of the data in one pytorch tensor (OUTPUT_SAVE).

    The data for each scene  is first turned into an 'image-like' tensor
    (meaning that each value correspond to a pixel) called Output.
    This tensor has shape 2 x XL x YL. Output[0] is a XL x YL
    tensor the carries information only of the emtpy drops, whilst
    Output[1] is a XL x YL tensor that carries information only for the
    drops containing cells.

    For simplicity, this tensor is then reshaped as line vector and stored
    in OUTPUT_SAVE. Each line of OUTPUT_SAVE corresponds to one scene's output.

    :param N_Img: Total number of images (int = 4859).
    :param XL: Number of pixels in the X axis (int = 336).
    :param YL: Number of pixels in the Y axis (int = 192).
    :return: OUTPUT_SAVE: The tensor storing all of the outputs (th.tensor).
             Valid_idx: A list containing the the valid scene indexes (list).
    """

    # Finding the valid scene values:
    Valid_idx = []  # The valid images
    N_Valid = N_Img  # Total amount of valid images
    Names = os.listdir()

    # for N in range(N_Img):  # from 0 to N_Img
    for N in range(len(Names)):  # from 0 to N_Img
        filename = Names[N]  # Defines the file's name
        match = re.search(r'scene\d{5}.txt', filename) # Checking for scenees
        if match == None: # If it isn't a scene
            continue # Skip it
        else:       # Otherwise:
            Valid_idx.append(filename)  # Add it to the valid list

    OUTPUT_SAVE = th.zeros(N_Valid, XL * YL * 2)  # Tensor for storing all of the outputs

    for N in range(len(Valid_idx)): # For the valid files:
        RawData = []
        filename = Valid_idx[N] # The valid name
        f = open(filename, 'r')  # Opens the file (now guaranteed to work)
        for line in f:
            line = line.strip()
            result = line.split(" ")
            RawData.append(result)
        f.close()
        # Output Tensor
        Output = th.zeros(2, XL, YL)  # Output[0]: Drop_0 data, Output[1]: Drop_1 data

        for i in range(len(RawData)):
            Drop_Type = int(RawData[i][0])  # Either 0 or 1, indicating the Drop Type

            xc = round(float(RawData[i][1]) * XL)  # X coordinate of the center of the rectangle
            yc = round(float(RawData[i][2]) * YL)  # Y coordinate of the center of the rectangle
            W = round(float(RawData[i][3]) * XL)  # Width (pixels)
            H = round(float(RawData[i][4]) * YL)  # Height (pixels)

            W2 = round(W / 2)  # Half of the width
            H2 = round(H / 2)  # Half of the height

            Output[Drop_Type, xc - W2:(xc + W2) + 1, yc - H2:(yc + H2) + 1] = 1  # Setting the drops as 1

        OUTPUT_SAVE[N, :] = Output.view(1, -1)  # Saving the final tensor as a vector

    return OUTPUT_SAVE, Valid_idx


def PLOT_TEST(OUTPUT_SAVE, Valid_idx, N, XL=336, YL=192):
    """
    This functions serves to plot the output tensor created with the READER function.
    This serves solely for testing.

    :param OUTPUT_SAVE: The output tensor that stores all of the outputs (th.tensor).
    :param N: The number of the scene
    :return: None
    """
    # Dictionary for translating a scene's number to a line of the output tensor:
    test_file = FILENAME_TXT(N) # The file name associated to scene N
    IDX_dict = {Valid_idx[i]: i for i in range(len(Valid_idx))} # Dictionary connecting filename and
    # the line in the OUTPUT_SAVE matrix.
    Test = OUTPUT_SAVE[IDX_dict[test_file], :]  # The output we want to test
    Test = Test.view(2, XL, YL)  # Turning it into the image's shape

    plt.matshow(np.transpose(Test.detach().numpy()[0]))  # Plotting for the Drop_0 type
    plt.show()
    plt.matshow(np.transpose(Test.detach().numpy()[1]))  # Plotting for the Drop_1 type
    plt.show()

    return None


OUT, IDX = READER()  # Just to show how to use it
#PLOT_TEST(OUT,IDX,1625)     # Plotting for scene nb 1625