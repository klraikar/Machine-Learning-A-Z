# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:33:08 2019

@author: Acer
"""

'''

                            Online Python Compiler.
                Code, Compile, Run and Debug python program online.
Write your code in this editor and press "Run" button to execute it.

'''

from itertools import combinations

def solution(L,X):
    total = 0
    for i in L:
        if prime(i) == 1:
            total = total + i 
    print(total)
    
    l_diff = []
    for j in L:
        l_diff.append(abs(j-total))
    print(l_diff)
        
    l_nonprime = []
    for k in l_diff:
        if prime(k) == -1:
            l_nonprime.append(k)
            
    if len(l_nonprime) == 0:
        return -1
    else:
        l_nonprime_sort = sorted(l_nonprime)
        print(l_nonprime_sort)
        
    r = len(l_nonprime)    
    perm = list(combinations(l_nonprime_sort,r))
    print(perm)

    sum_less_than_X = []
    for sub_arr in perm:
        if sum(sub_arr) < X :
            sum_less_than_X.append(sub_arr)
            print(sub_arr)
    print(len(sum_less_than_X))
    
    
def prime(num):
    if num > 1:
        for i in range(2,num):
            if (num % i ) == 0:
                return -1
                break
        else:
            return 1
    else:
        return -1
        
        
L = [13, 18, 1, 3, 4, 5, 50, 29, 30, 41]
X = 200
solution(L,X)