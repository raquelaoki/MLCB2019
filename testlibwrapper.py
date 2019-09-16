# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:47:29 2019

@author: raoki
"""

import ctypes

testlib = ctypes.CDLL('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\testlib.dll')
testlib.myprint()