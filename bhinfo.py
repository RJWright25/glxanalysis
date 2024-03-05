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


# bhprocessing.py
# This file contains the functions to process the black hole data from KETJU simulations.

import os
import numpy as np
import pandas as pd

# This function is used to postprocess the blackhole details files. (credit: Shihong Liao reprocess.py)
def postprocess_bhdata(path=None):
    """
    Postprocesses the black hole details files.

    Parameters:
    -----------
    simulation: simulation object
        The simulation object for which the black hole details are to be postprocessed.
    path: str
        The path to the directory containing the black hole details files.
    """

    # if no path, get from simulation
    if not path:
        print('No path given. Exiting...')
        return None

    # Specify file path and target BH ids
    fileNum = 0
    fileName = f"{path}/blackhole_details/blackhole_details_{fileNum}.txt"

    while(os.path.isfile(fileName)):
        fileNum += 1
        fileName = f"{path}/blackhole_details/blackhole_details_{fileNum}.txt" 
    print('Total files found:', fileNum)

    BHDetails = {}
    # Load files
    for file_index in list(range(fileNum))[:20]:
        if file_index % 10 == 0:
            print('Processing file:', file_index+1, '/', fileNum)

        fileName = f"{path}/blackhole_details/blackhole_details_{file_index}.txt"
        data = pd.read_csv(fileName, header=None, delimiter=" ")
        # data[0] contains "BH=ID". Find the unique BH IDs in this file:
        BHIDsInFile = data[0].str.extract('BH=(\d+)').values.flatten()
        BHIDsInFile = [int(BHID) for BHID in BHIDsInFile if np.isfinite(np.float32(BHID))]
        BHIDsInFile = np.unique(BHIDsInFile)
        BHNum= len(BHIDsInFile)
        
        for ibh in range(BHNum):
            BHID=BHIDsInFile[ibh]
            select_data = data.loc[data[0].str.contains(f'BH={BHID}'),:]
            if not f'{BHID}' in BHDetails:
                BHDetails[f'{BHID}'] = select_data
            else:
                BHDetails[f"{BHID}"] = [BHDetails[f"{BHID}"],select_data]
                BHDetails[f"{BHID}"] = pd.concat(BHDetails[f"{BHID}"],ignore_index=True)

    #Remove string columns
    for ibh in range(BHNum):
        BHDetails[f"{BHIDs[ibh]}"] = BHDetails[f"{BHIDs[ibh]}"].drop(columns=[0])
    
    print(BHDetails[str(BHIDs[0])])

        
    #check first value of each column to see if it is a nan
    BHIDs = np.array(list(BHDetails.keys()))
    BHIDs = np.array([int(BHIDs[ibh]) for ibh in range(BHNum)])
    for ibh in range(len(BHIDs)):
        BHDetails[f"{BHIDs[ibh]}"]=BHDetails[f"{BHIDs[ibh]}"].dropna(axis=1,how='all')

    # Get the number of BHs
    BHNum = len(BHDetails)
    print('BH number = ', BHNum)

    #print number of columns
    print('Number of columns:',BHDetails[str(BHIDs[0])].shape[1])

    print()

    # Sort according to time
    for ibh in range(BHNum):
        BHDetails[f"{BHIDs[ibh]}"] = BHDetails[str((BHIDs[ibh]))].sort_values(by=[1])
        BHDetails[f"{BHIDs[ibh]}"].reset_index(inplace=True,drop=True)
        
    # Save files
    if not os.path.exists('blackhole_details_post_processing'):
        os.mkdir('blackhole_details_post_processing')
    else:
        # Remove all files in the directory
        files = os.listdir('blackhole_details_post_processing')
        for file in files:
            os.remove(f'blackhole_details_post_processing/{file}')

    for ibh in range(BHNum):
        fname = f'blackhole_details_post_processing/BH_{BHIDs[ibh]}.txt'
        BHDetails[str(BHIDs[ibh])].to_csv(fname, sep=' ', index=False, header=False)

    return BHDetails


# This function is used to read the black hole details from a file
def read_bhdata(simulation,path=None,bhids=None,subsample=1):
    """
    Reads the black hole details from a file.

    Parameters:
    -----------
    simulation: simulation object
        The simulation object for which the black hole details are to be read.
    path: str
        The path to the directory containing the black hole details files.
    bhids: lsit
        The IDs of the black hole to read (may not be all bhs).
    subsample: int
        The subsampling factor to use when reading the data.
    
    Returns:
    -----------
    bhdata : dict
        A dictionary containing a pandas dataframe of the data for each black hole.
        Keys are the black hole IDs.

    """
    if not path:
        path=simulation.snapshots[0].snapshot_file.split('/')[:-1]
        path='/'.join(path)+'/blackhole_details_post_processing/'
        if os.path.exists(path):
            print(f'Using path {path} to read black hole details...')
        else:
            print('No path found. Exiting...')
            return None
    
    #find all the files in the directory
    bhfiles=np.array([path+fname for fname in os.listdir(path) if 'BH' in fname])

    #cull the list if not all BHs are requested
    if bhids:
        bhfiles=np.array([fname for fname in bhfiles if int(fname.split('/BH_')[-1].split('.txt')[0]) in bhids])
    else:
        bhids=np.array([int(fname.split('/BH_')[-1].split('.txt')[0]) for fname in bhfiles])

    #sort by bhid
    bhids=bhids[np.argsort(bhids)]
    bhfiles=bhfiles[np.argsort(bhids)]

    #columns
    fpath=path+f'/BH_{str(int(bhids[0]))}.txt'
    bhdata_ibh=np.loadtxt(fpath,dtype=str)[::subsample,1:].astype(float)
    numcol=bhdata_ibh.shape[-1]
    columns=np.array(['Time','bh_M','bh_Mdot','rho','cs','gas_Vrel_tot','Coordinates_x','Coordinates_y','Coordinates_z','V_x','V_y','V_z','gas_Vrel_x','gas_Vrel_y','gas_Vrel_z','Flag_binary','companion_ID','bh_hsml'])
    columns=columns[:numcol]

    #initialize the output
    bhdata={}

    #read the data
    for bhid in bhids:
        print(f'Reading black hole details for BHID = {bhid}...')
        fpath=path+f'/BH_{str(int(bhid))}.txt'
        bhdata_ibh=np.loadtxt(fpath,dtype=str)[::subsample,1:].astype(float)

        bhdata_ibh=pd.DataFrame(bhdata_ibh,columns=columns)
        bhdata_ibh['BH_ID']=np.ones(bhdata_ibh.shape[0])*int(bhid)

        #if cosmo, convert afac to time
        if simulation.snapshot_type=='cosmo':
            bhdata_ibh['Time']=simulation.cosmo.age(1/bhdata_ibh['Time']-1)

        #convert to physical units
        bhdata_ibh['bh_M']=bhdata_ibh['bh_M']*1e10/simulation.hubble
        bhdata_ibh['bh_Mdot']=bhdata_ibh['bh_Mdot']#don't think this needs to be converted
        for key in [f'Coordinates_{x}' for x in 'xyz']:
            bhdata_ibh[key]=bhdata_ibh[key]/simulation.hubble

        #now add closest snap index from the main simulation to the BH data
        bhdata_ibh['isnap']=np.zeros(bhdata_ibh.shape[0])
        for isnap,snapshot in enumerate(simulation.snapshots):
            bhdata_ibh.loc[bhdata_ibh['Time'].values>=snapshot.time,'isnap']=isnap

        #add to the output
        bhdata[bhid]=bhdata_ibh
    
    return bhdata

