import os
import numpy as np
import matplotlib.pyplot as plt
import re

dir_labels = "C:\\Users\\Viacheslav RUSANOV\\Desktop\\Test"

H = 192
W = 336

class Drop_label:
    def __init__(self, type, x, y, width, hight,):
        self.xmin = int((x - width / 2) * W)
        self.ymin = int((y - hight / 2) * H)
        self.xmax = int((x + width / 2) * W)
        self.ymax = int((y + hight / 2) * H)
        self.type = type

class Labeled_image:
    def __init__(self):
        self.obects = []

    def append(self, drop_label):
        self.obects.append(drop_label)

    def matrix(self):
        matrix = np.zeros((H, W))

        for obj in self.obects:
            matrix[obj.ymin:obj.ymax, obj.xmin:obj.xmax] = 255

        return matrix

    def show(self):
        plt.imshow(self.matrix())
        plt.show()


def get_labels():
    names = os.listdir(dir_labels)
    label_imgs = []
    for name in names:
        fullname = os.path.join(dir_labels, name)
        match = re.search(r'scene\d{5}.txt', fullname)
        if match != None:
            file = open(fullname)
            label_img = Labeled_image()
            for line in file:
                line = line.strip()
                result = line.split(" ")
                object = Drop_label(int(result[0]), float(result[1]), float(result[2]), float(result[3]), float(result[4]))
                label_img.append(object)
            file.close()
            label_imgs.append(label_img)

    return label_imgs


if __name__ == '__main__':
    print("nope")

