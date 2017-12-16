# coding=utf-8

#!/usr/bin/python

#
# Implements 8-connectivity connected component labeling
# 
# Algorithm obtained from "Optimizing Two-Pass Connected-Component Labeling 
# by Kesheng Wu, Ekow Otoo, and Kenji Suzuki
#
from PIL import Image
from PIL import ImageDraw
import collections

import operator
import numpy as np
import sys
import math, random
from itertools import product
from ufarray import *
Image.MAX_IMAGE_PIXELS = 1000000000



def run(img):
    data = img.load()
    width, height = img.size
 
    # Union find data structure
    uf = UFarray()
 
    #
    # First pass
    #
 
    # Dictionary of point:label pairs
    labels = {}
 
    for y, x in product(range(height), range(width)):
 
        #
        # Pixel names were chosen as shown:
        #
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e |   |
        #   -------------
        #   |   |   |   |
        #   -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #
 
        # If the current pixel is white, it's obviously not a component...
        if data[x, y] == 255:
            pass
 
        # If pixel b is in the image and black:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and data[x, y-1] == 0:
            labels[x, y] = labels[(x, y-1)]
 
        # If pixel c is in the image and black:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x+1 < width and y > 0 and data[x+1, y-1] == 0:
 
            c = labels[(x+1, y-1)]
            labels[x, y] = c
 
            # If pixel a is in the image and black:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and data[x-1, y-1] == 0:
                a = labels[(x-1, y-1)]
                uf.union(c, a)
 
            # If pixel d is in the image and black:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and data[x-1, y] == 0:
                d = labels[(x-1, y)]
                uf.union(c, d)
 
        # If pixel a is in the image and black:
        #    We already know b and c are white
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and data[x-1, y-1] == 0:
            labels[x, y] = labels[(x-1, y-1)]
 
        # If pixel d is in the image and black
        #    We already know a, b, and c are white
        #    so simpy assign d's label to e
        elif x > 0 and data[x-1, y] == 0:
            labels[x, y] = labels[(x-1, y)]
 
        # All the neighboring pixels are white,
        # Therefore the current pixel is a new component
        else: 
            labels[x, y] = uf.makeLabel()
 
    #
    # Second pass
    #
 
    uf.flatten()
 
    colors = {}

    # Image to display the components in a nice, colorful way
    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()

    for (x, y) in labels:
 
        # Name of the component the current point belongs to
        component = uf.find(labels[(x, y)])

        # Update the labels with correct information
        labels[(x, y)] = component

        # Associate a random color with this component 
        if component not in colors: 
            colors[component] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))

        # Colorize the image
        outdata[x, y] = colors[component]

# ============================================================================================================================

    return (labels, output_img)


def main():
    # Open the image
    img = Image.open("C:\Users\student\PyCharm\Masterarbeit\output\Test\TEST (13-11-17)_inv.png")

    # Threshold the image, this implementation is designed to process b+w
    # images only
    img = img.point(lambda p: p > 190 and 255)
    img = img.convert('1')

    # labels is a dictionary of the connected component data in the form:
    #     (x_coordinate, y_coordinate) : component_id
    #
    # if you plan on processing the component data, this is probably what you
    # will want to use
    #
    # output_image is just a frivolous way to visualize the components.

    (labels, output_img) = run(img)
    #output_img.show()

######################################################################################
    sorted_x = sorted(labels.items(), key=operator.itemgetter(1))


    str_labels = str(sorted_x)      # Dictionary in String umwandeln, unnötige Zeichen entfernen
    str_labels = str_labels.replace("{", "[")
    str_labels = str_labels.replace("}", "]")
    str_labels = str_labels.replace("(", "")
    str_labels = str_labels.replace(")", "")
    str_labels = str_labels.replace(":", ",")
    # print str_labels[:100]

    int_labels = eval(str_labels)   # Rücktransformieren in integer-Liste

    print len(int_labels), int_labels[:1000]


    i = 0
    key = []
    value = []
    #int_labels2 = list(int_labels)
    while i < len(int_labels)/3:
        key.append(int_labels[3*i])         # x-Werte (keys)
        key.append(int_labels[3 * i + 1])   # y-Werte (keys)
        value.append(int_labels[3 * i + 2]) # values
        i = i + 1

    print "key: ", len(key), key[:100]
    print "value: ", len(value), value[:100]


    counter = collections.Counter(value)    # Zählen der Häufigkeit der "values"
    counter_values = counter.values()

    print "counter_values: ", len(counter_values), counter_values
    print sum(counter_values)*2


################################################################################
# Selektion
################################################################################
# Aussortieren von "keys" und "values", die zu groß/klein sind
# key2 = relevante "keys"
# value2 = zu key2 zugehörige relevante values


    i = 0
    k = 0
    v = 0
    key2 = []
    key3 = []
    value2 = []
    value3 = []
    help = []

    while i < len(counter_values):
        if counter_values[i] >= 35 and counter_values[i] <= 70:
            value2.append(counter_values[v]*2)
            v = v + 1

            while k < counter_values[i]*2 + sum(help)*2:
                key2.append(key[k])
                k = k + 1

        else:
            value3.append(counter_values[v]*2)
            v = v + 1

            while k < counter_values[i]*2 + sum(help)*2:
                key3.append(key[k])
                k = k + 1

        # print k
        # print counter_values[i]*2 + sum(help)*2
        help.append(counter_values[i])

        i = i + 1



    print "key2: ", len(key2), key2[:100]
    # print "key3: ", len(key3), key3[:100]
    print "value2: ", len(value2), value2[:100]
    # print "value3: ", len(value3), value3[:100]

    # print "k: ", k
    # print "i: ", i

    # print counter_values[0]*2

    # print len(help), help

    # print sum(value2)



if __name__ == "__main__":
    main()
