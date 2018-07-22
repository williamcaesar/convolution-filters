#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
from cv2 import cv2


def add_zero(matrix):
    """Create rows of zeroes and put them on begining or the tail of the array."""
    if len(matrix.shape) == 2:
        matrix = numpy.insert(matrix, 0, 0, axis=1)
        matrix = numpy.insert(matrix, matrix.shape[1], 0, axis=1)
        row01 = numpy.zeros((matrix.shape[1]), numpy.uint8)
        matrix = numpy.insert(matrix, 0, row01, axis=0)
        matrix = numpy.insert(matrix, matrix.shape[0], row01, axis=0)

    if len(matrix.shape) == 3:
        pixel = numpy.zeros((matrix.shape[2]), numpy.uint8)
        matrix = numpy.insert(matrix, 0, pixel, axis=1)
        matrix = numpy.insert(matrix, matrix.shape[1], pixel, axis=1)
        row = numpy.zeros((1, matrix.shape[1], 3), numpy.uint8)
        matrix = numpy.insert(matrix, 0, row, axis=0)
        matrix = numpy.insert(matrix, matrix.shape[0], row, axis=0)

    return matrix


def low_pass(image):
    """Low pass filter."""
    image = add_zero(image)
    if len(image.shape) == 2:
        result = numpy.zeros((image.shape[0]-2, image.shape[1]-2), numpy.uint8)
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                row01 = int(image[row-1, column-1]) + \
                              int(image[row-1, column]) + \
                                  int(image[row-1, column + 1])
                row02 = int(image[row, column-1]) + \
                    int(image[row, column]) + int(image[row, column + 1])
                row03 = int(image[row+1, column-1]) + \
                    int(image[row+1, column]) + \
                    int(image[row+1, column + 1])
                result[row-1, column-1] = (row01 + row02 + row03)/9
    else:
        result = numpy.zeros(
            (image.shape[0]-2, image.shape[1]-2, image.shape[2]), numpy.uint8)
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                for channel in range(0, image.shape[2]):
                    row01 = int(image[row-1, column-1, channel]) + int(
                        image[row-1, column, channel]) + \
                        int(image[row-1, column+1, channel])

                    row02 = int(image[row, column-1, channel]) + \
                        int(image[row, column, channel]) + \
                        int(image[row, column+1, channel])

                    row03 = int(image[row+1, column-1, channel]) + int(
                        image[row+1, column, channel]) + int(image[row+1,
                                                                   column+1,
                                                                   channel])

                    result[row-1, column-1,
                           channel] = (row01 + row02 + row03)/9

    print(" low pass = ", result.shape)
    return result


def median(image):
    """Median filter."""
    print("Result dimensions ", image.shape)
    image = add_zero(image)

    if len(image.shape) == 2:
        result = numpy.zeros((image.shape[0]-2, image.shape[1]-2), numpy.uint8)
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                roll = [image[row-1, column-1], image[row-1, column],
                        image[row-1, column + 1], image[row, column-1],
                        image[row, column],
                        image[row, column + 1],
                        image[row+1, column-1],
                        image[row+1, column],
                        image[row+1, column + 1]]
                row01 = int(image[row-1, column-1]) + \
                    int(image[row-1, column]) + \
                    int(image[row-1, column + 1])
                row02 = int(image[row, column-1]) + \
                    int(image[row, column]) + int(image[row, column + 1])
                row03 = int(image[row+1, column-1]) + \
                    int(image[row+1, column]) + \
                    int(image[row+1, column + 1])
                result[row-1, column-1] = (row01 + row02 + row03)/9
                roll.sort()  # Ordena a lista
                result[row-1, column-1] = roll[4]
    else:
        result = numpy.zeros(
            (image.shape[0]-2, image.shape[1]-2, image.shape[2]), numpy.uint8)
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                for channel in range(0, image.shape[2]):
                    roll = [image[row-1, column-1, channel],
                            image[row-1, column, channel],
                            image[row-1, column+1, channel],
                            image[row, column-1, channel],
                            image[row, column, channel],
                            image[row, column+1, channel],
                            image[row+1, column-1, channel],
                            image[row+1, column, channel],
                            image[row+1, column+1, channel]
                            ]
                    roll.sort()
                    result[row-1, column-1, channel] = roll[4]
    return result


def gaussian(image):
    """Gussian filter."""
    print("Result dimensions ", image.shape)
    image = add_zero(image)
    mask = (1, 2, 1, 2, 4, 2, 1, 2, 1)
    if len(image.shape) == 2:
        result = numpy.zeros((image.shape[0]-2, image.shape[1]-2), numpy.uint8)
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                row01 = int(image[row-1, column-1]) * \
                 mask[0] + int(image[row-1, column]) * \
                 mask[1] + int(image[row-1, column + 1]) * mask[2]

                row02 = int(image[row, column-1]) * \
                    mask[3] + int(image[row, column]) * \
                    mask[4] + int(image[row, column + 1]) * mask[5]

                row03 = int(image[row+1, column-1]) * \
                    mask[6] + int(image[row+1, column]) * mask[6] + \
                    int(image[row+1, column + 1]) * mask[8]
                result[row-1, column-1] = (row01 + row02 + row03)/16
    else:
        result = numpy.zeros(
            (image.shape[0]-2, image.shape[1]-2, image.shape[2]), numpy.uint8)
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                for channel in range(0, image.shape[2]):
                    row01 = int(image[row-1, column-1, channel]) * mask[0] + \
                        int(image[row-1, column, channel]) * mask[1] + \
                        int(image[row-1, column+1, channel]) * mask[2]

                    row02 = int(image[row, column-1, channel]) * mask[3] + \
                        int(image[row, column, channel]) * mask[4] + \
                        int(image[row, column+1, channel]) * mask[5]

                    row03 = int(image[row + 1, column-1, channel]) * \
                        mask[6] + int(image[row+1, column, channel])*mask[7] + \
                        int(image[row+1, column+1, channel])*mask[8]

                    result[row-1, column-1,
                           channel] = (row01 + row02 + row03)/16

    return result


def high_pass(image):
    """High pass filter."""
    print("Result dimensions ", image.shape)
    image = add_zero(image)

    mask = (-1, -1, -1, -1, 9, -1, -1, -1, -1)
    # mask = ( 0, -1, 0, -1, 5, -1, 0, -1, 0)
    if len(image.shape) == 2:
        result = numpy.zeros((image.shape[0]-2, image.shape[1]-2), numpy.uint8)
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                row01 = int(image[row-1, column-1])*mask[0] + \
                    int(image[row-1, column])*mask[1] + \
                    int(image[row-1, column + 1])*mask[2]

                row02 = int(image[row, column-1])*mask[3] + \
                    int(image[row, column])*mask[4] + \
                    int(image[row, column + 1])*mask[5]

                row03 = int(image[row+1, column-1])*mask[6] + \
                    int(image[row+1, column])*mask[6] + \
                    int(image[row+1, column + 1])*mask[8]

                result[row-1, column-1] = (row01 + row02 + row03)/9
    else:
        result = numpy.zeros(
            (image.shape[0]-2, image.shape[1]-2, image.shape[2]), numpy.uint8)
        for row in range(1, image.shape[0]-1):
            for column in range(1, image.shape[1]-1):
                for channel in range(0, image.shape[2]):
                    row01 = int(image[row-1, column-1, channel])*mask[0] + \
                        int(image[row-1, column, channel])*mask[1] + \
                        int(image[row-1, column+1, channel])*mask[2]

                    row02 = int(image[row, column-1, channel])*mask[3] + \
                        int(image[row, column, channel])*mask[4] + \
                        int(image[row, column+1, channel])*mask[5]

                    row03 = int(image[row+1, column-1, channel])*mask[6] + \
                        int(image[row+1, column, channel])*mask[7] + \
                        int(image[row+1, column+1, channel])*mask[8]

                    result[row-1, column-1,
                           channel] = (row01 + row02 + row03)/9

    return result


def show(image):
    """Show a given image till ENTER is pressed."""
    cv2.imshow('press ENTER to close', image)
    cv2.waitKey(0)


if (__name__ == '__main__'):

    image = cv2.imread('images/sapo.png')
    show(image)
    show(high_pass(image))
    show(low_pass(image))
    show(median(image))
    show(high_pass(image))
    show(gaussian(image))
