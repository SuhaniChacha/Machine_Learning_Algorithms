# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def fact(n):
    if n == 0 or n==1:
        return 1
    else:
        return n* fact(n-1)
print(fact(int(input())))