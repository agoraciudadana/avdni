#!/usr/bin/python
#
# AVdni is recogniter of ocr dni for agoravoting.
# Copyright (C) 2014 Victor Ramirez de la Corte <virako at wadobo dot com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
import numpy as np
from subprocess import call


IMAGE = 'imgs/dni.jpg'
TMP = 'tmp'
DEBUG = False


def filter_img(filename):
    orig_img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img = cv2.blur(orig_img, (3, 3))
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 11, 3)
    fth = filename.replace('.', '_canny.')
    cv2.imwrite(fth, th)
    return fth


def filter_txt(txt):
    ini = txt.find('IDESP')
    fin = len(txt) - txt[::-1].find('<')
    return txt[ini:fin]


def main():
    # TODO: get dni and change perspective
    fil = filter_img(IMAGE)
    call('tesseract %s %s -l spa' % (fil, fil), shell=True)
    tmp = open(fil + '.txt', 'r')
    txt = tmp.read()
    tmp.close()
    print filter_txt(txt)

if __name__ == "__main__":
    main()
