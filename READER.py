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

def READER(N_min=1,N_max=4859,Missing = [0,1623,1626,1632,1648,1650,1655,1660,1670,1690,1695,1708,1710,1717,1719],N_Img=4859):
    """ This function serves to interpret .txt files for the project
    and to save all of the data in one pytorch tensor (OUTPUT_SAVE).

    The data for each scene  is first turned into an 'image-like' tensor
    (meaning that each value correspond to a pixel) called Output.
    This tensor has shape 2 x 336 x 192. Output[0] is a 336 x 192
    tensor the carries information only of the emtpy drops, whilst
    Output[1] is a 336 x 192 tensor that carries information only for the
    drops containing cells.

    For simplicity, this tensor is then reshaped as line vector and stored
    in OUTPUT_SAVE. Each line of OUTPUT_SAVE corresponds to one scene's output.

    OUTPUT_SAVE[i] is the output of scene i.
    This means that OUTPUT_SAVE[0] and OUTPUT_SAVE[j] for j in the missing scenes is zero.
    This must be taken into account for the training (ignore the missing scenes).

    :param N_min: First scene to take into account (int).
    :param N_max: Last scene to take into account (int).
    :param Missing: Missing scenes. For some reason there are missing scenes (list of int's).
    :param N_Img: Total number of images (4859).
    :return: OUTPUT_SAVE: The tensor storing all of the outputs (th.tensor).
    """

    #N_Img = 4859                                # Total number of images
    OUTPUT_SAVE = th.zeros(N_Img,336*192*2)     # Tensor for storing all of the outputs
    #% We may have to adapt this size/shape later on, when implementing batch training. %#

    # The following list groups the missing scenes so far
    #Missing = [1623,1626,1632,1648,1650,1655,1660,1670,1690,1695,1708,1710,1717,1719]

    for N in range(N_min,N_max+1):      # May have to be adapted according to testing
        if N in Missing:    # If N corresponding to a missing scene
            continue        # Then continue to next value of N

        filename = FILENAME_TXT(N)              # Defines the file's name
        f = open(filename,'r')                  # Open the file
        RawData = f.readlines()                 # Reads the raw data from the file
        f.close()

        # Output Tensor
        Output = th.zeros(2,336,192)            # Output[0]: Drop_0 data, Output[1]: Drop_1 data

        for i in range(len(RawData)):

            Drop_Type = int(RawData[i][0])   # Either 0 or 1, indicating the Drop Type

            xc = round(float(RawData[i][2:10])*336)     # X coordinate of the center of the rectangle
            yc = round(float(RawData[i][11:19])*192)    # Y coordinate of the center of the rectangle
            W = round(float(RawData[i][20:28])*336)     # Width (pixels)
            H = round(float(RawData[i][29:37])*192)     # Height (pixels)

            W2 = round(W/2)     # Half of the width
            H2 = round(H/2)     # Half of the height

            Output[Drop_Type,xc-W2:(xc+W2)+1,yc-H2:(yc+H2)+1] = 1   # Setting the drops as 1

        OUTPUT_SAVE[N,:] = Output.view(1,-1)    # Saving the final tensor as a vector

    return OUTPUT_SAVE

def PLOT_TEST(OUTPUT_SAVE,N_min=1,N_max=4858):
    """
    This functions serves to plot the output tensor created with the READER function.
    This serves solely for testing.
    The first and last plots should be empty (purple).
    This will break if N_max = N_Img = 4859 because the last plot is for line N_max+1
    which is impossible if N_max is already the tensor's limit (len(OUTPUT_SAVE[0 or 1]).

    It will plot an empty plot (scene N_min-1)
    Then a plot for the Drop_0 class for scene N_min
    Then a plot for the Drop_1 class for scene N_min
    Then a plot for the Drop_0 class for scene N_max
    Then a plot for the Drop_1 class for scene N_max
    Then another empty plot (scene N_max+1)

    :param OUTPUT_SAVE: The output tensor that stores all of the outputs (th.tensor).
    :param N_min: The minimum scene value to check (int).
    :param N_max: The maximum scene value to check (int).
    :return: None
    """
    Test1 = OUTPUT_SAVE[N_min-1,:]
    Test2 = OUTPUT_SAVE[N_min,:]
    Test3 = OUTPUT_SAVE[N_max,:]
    Test4 = OUTPUT_SAVE[N_max+1,:]

    Test1 = Test1.view(2,336,192)
    Test2 = Test2.view(2,336,192)
    Test3 = Test3.view(2,336,192)
    Test4 = Test4.view(2,336,192)

    plt.matshow(np.transpose(Test1.detach().numpy()[0]))
    plt.show()
    plt.matshow(np.transpose(Test2.detach().numpy()[0]))
    plt.show()
    plt.matshow(np.transpose(Test2.detach().numpy()[1]))
    plt.show()
    plt.matshow(np.transpose(Test3.detach().numpy()[0]))
    plt.show()
    plt.matshow(np.transpose(Test3.detach().numpy()[1]))
    plt.show()
    plt.matshow(np.transpose(Test4.detach().numpy()[1]))
    plt.show()

    return None

OUT = READER(1620,1720)     # Just to show how to use it - here for scenes 1620 - 1720, that I labelled
PLOT_TEST(OUT,1620,1720)    # Plotting