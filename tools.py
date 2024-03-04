
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


# This function is used to postprocess the blackhole details files. (credit: Shihong Liao reprocess.py)
def postprocess_bhinfo(filePath):

    # Specify file path and target BH ids
    fileNum = 0
    fileName = "%sblackhole_details/blackhole_details_%d.txt" % (filePath, fileNum)
    while(os.path.isfile(fileName)):
        fileNum += 1
        fileName = "%sblackhole_details/blackhole_details_%d.txt" % (filePath, fileNum)

    print('Total files found:', fileNum)

    BHDetails = {}
    # Load files
    for file_index in range(fileNum):
        if file_index % 100 == 0:
            print(file_index)
        fileName = "%sblackhole_details/blackhole_details_%d.txt" % (filePath, file_index)
        data = pd.read_csv(fileName, header=None, delimiter=" ")
        # data[0] contains "BH=ID". Find the unique BH IDs in this file:
        BHIDsInFile = data[0].str.extract('BH=(\d+)').astype(int).values.flatten()
        BHIDsInFile = np.unique(BHIDsInFile)
        BHNum= len(BHIDsInFile)
        for i in range(BHNum):
            select_data = data.loc[data[0].str.contains('BH=%d' % BHIDsInFile[i]),:]
            if file_index == 0:
                BHDetails['BH%d' % (BHIDsInFile[i])] = select_data
            else:
                BHDetails['BH%d' % (BHIDsInFile[i])] = [BHDetails['BH%d' % (BHIDsInFile[i])],select_data]
                BHDetails['BH%d' % (BHIDsInFile[i])] = pd.concat(BHDetails['BH%d' % (BHIDsInFile[i])])

    # Get the number of BHs
    BHNum = len(BHDetails)
    print('BH number = ', BHNum)

    # Get the BH IDs
    BHIDs = np.array(list(BHDetails.keys()))
    BHIDs = np.array([int(BHIDs[i][2:]) for i in range(BHNum)])

    # Get the number of columns
    col_num = len(BHDetails['BH%d' % (BHIDs[0])]['data'].columns)
    print('column number = ', col_num)

    # Sort according to time
    for i in range(BHNum):
        BHDetails['BH%d' % (BHIDs[i])] = BHDetails['BH%d' % (BHIDs[i])]['data'].sort_values(by=[col_num-1])

    # Save files
    if not os.path.exists(filePath + 'blackhole_details_post_processing'):
        os.makedirs(filePath + 'blackhole_details_post_processing')
    for i in range(BHNum):
        fname = './BH_%d.txt' % (BHIDs[i])
        BHDetails['BH%d' % (BHIDs[i])].to_csv(fname, sep=' ', index=False, header=False)


