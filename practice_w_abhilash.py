#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:07:43 2018

@author: whitneyreiner
"""
#make a linked list from scratch

# =============================================================================
# Q1 Implement an algorithm to determine if a string has all unique characters. 
# What if you cannot use additional built-in data structures other than 
# strings?
# 
# =============================================================================
# is everything good in code, read next line in code
#continue means skip code
x = 'insight is lit'
z = []
for i in x:
	if x is not :
x= x.strip() #unnecessary

x = 'insight is lit'
z = []
for i in x:
	if i not in z:
    z.append(i)
  else:
    continue

if len(x) == len(z):
  print('yep')
  
  
  
uni = [i for i in x if i not in x]


#go through one array no need for range or len
# range(len())
if access more than one array and are mushed together and need to maintain index order :
(if you forget str are iterable or you care about the explicit length of the string)
#Doesn't know what the index is if iterating through the string, so if it matters then you need to do range(len())

#permutation- case and space sensitive