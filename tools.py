
##  .d8888b.           888                                        d8888                   888                   d8b          
## d88P  Y88b          888                                       d88888                   888                   Y8P          
## 888    888          888                                      d88P888                   888                                
## 888         8888b.  888  8888b.  888  888 888  888          d88P 888 88888b.   8888b.  888 888  888 .d8888b  888 .d8888b  
## 888  88888     "88b 888     "88b `Y8bd8P' 888  888         d88P  888 888 "88b     "88b 888 888  888 88K      888 88K      
## 888    888 .d888888 888 .d888888   X88K   888  888        d88P   888 888  888 .d888888 888 888  888 "Y8888b. 888 "Y8888b. 
## Y88b  d88P 888  888 888 888  888 .d8""8b. Y88b 888       d8888888888 888  888 888  888 888 Y88b 888      X88 888      X88 
##  "Y8888P88 "Y888888 888 "Y888888 888  888  "Y88888      d88P     888 888  888 "Y888888 888  "Y88888  88888P' 888  88888P' 
##                                                888                                              888                       
##                                           Y8b d88P                                         Y8b d88P                       
##                                            "Y88P"                                           "Y88P"                        


# tools.py
# This file contains some useful functions that are used in the main functions.

import os
import numpy as np
import pandas as pd

from multiprocessing import Lock

# This function is used to print a string in a thread-safe manner.
def locked_print(string):
    """
    Prints a string in a thread-safe manner.

    Parameters:
    -----------
    string: str
        The string to be printed.
    
    Returns:
    -----------
    None (prints the string to the console)

    """

    print_lock = Lock()
    print_lock.acquire()
    print(string)
    print_lock.release()

# This function is used to split a list into nproc parts.
def split_list(lst,nproc):
    """
    Splits a list into nproc parts.
    
    Parameters:
    -----------
    lst: list
        The list to be split.
    nproc: int
        The number of processes to split the list into.

    Returns:
    -----------
    list (of lists)
        A list of nproc lists, each containing a subset of the original list.
    
    """
    return [lst[i::nproc] for i in range(nproc)]

