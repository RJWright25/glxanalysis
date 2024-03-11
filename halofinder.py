
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


# halofinder.py
# This file contains the function and tools to find haloes in a snapshot.

import os
import time
import numpy as np
import pandas as pd
import astropy.units as apy_units

from .tools import locked_print

# This function is used to find haloes in a snapshot.
def basic_halofinder(snapshot,delta=200,useminpot=False,verbose=True):

    """
    Basic halo finder for idealised/small hydro runs. Uses BH locations to find halo centres,
    then finds the minimum potential of the DM particles within a given radius of the BH.
    Returns a pandas dataframe with the properties of the halos found in the snapshot.

    Parameters
    ----------
    snapshot : snapshot object (or similar)
        Particle data object containing the snapshot data.
    delta : float
        Overdensity criteria for the halo finder.  
        Numeric value for overdensity.
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the halo finding.

    Returns
    ----------
    haloes : pandas dataframe
        Pandas dataframe containing the properties of the halos found in the snapshot.
        
    Output fields:
    ----------
    ID : int
        Halo ID number.
    x : float
        x position of the halo based on black hole location.
    y : float
        y position of the halo based on black hole location.
    z : float
        z position of the halo based on black hole location.
    xminpot : float
        x position of the halo based on (stellar) minimum potential.
    yminpot : float
        y position of the halo based on (stellar) minimum potential.
    zminpot : float
        z position of the halo based on (stellar) minimum potential. 
    vx : float
        x velocity of the halo based on mean velocity of central DM particles.
    vy : float
        y velocity of the halo based on mean velocity of central DM particles.
    vz : float
        z velocity of the halo based on mean velocity of central DM particles.
    Halo_M_{delta}c : float
        Mass of the halo for the specified overdensity.
    Halo_R_{delta}c : float
        Radius of the halo for the specified overdensity.


    """
    if verbose:

        print(f'===========================================================================================')
        print(f'Finding haloes in snapshot {snapshot.snapshot_file}...')
        print(f'===========================================================================================')
        print()        


    t0=time.time()
    cosmo=snapshot.cosmology

    #create a pandas dataframe to store the halo properties
    columns=['Time','isnap','ID','x','y','z','xminpot','yminpot','zminpot','vx','vy','vz',f'Halo_M_Crit{delta}',f'Halo_R_Crit{delta}']
    
    #initialize the dataframe with the number of BHs
    bhlocs=snapshot.get_particle_data(keys=['Coordinates','Velocities','Masses','ParticleIDs'],types=5)
    bhlocs.sort_values(by='ParticleIDs',ascending=True,inplace=True)
    bhlocs.reset_index(drop=True,inplace=True)
    numbh=bhlocs.shape[0]
    halo_output={column:np.zeros(bhlocs['Masses'].shape[0])+np.nan for column in columns}

    #for each "galaxy"
    for ibh in range(numbh):
        if verbose:
            print(f'Considering BH {ibh+1}/{numbh} (ID={bhlocs["ParticleIDs"].values[ibh]})...')

        ibh_row=bhlocs.iloc[ibh]
        #save the snapshot number and time
        halo_output['Time'][ibh]=snapshot.time
        halo_output['isnap'][ibh]=snapshot.snapshot_idx
        halo_output['ID'][ibh]=int(ibh_row['ParticleIDs'])
        halo_output['x'][ibh]=ibh_row['Coordinates_x']
        halo_output['y'][ibh]=ibh_row['Coordinates_y']
        halo_output['z'][ibh]=ibh_row['Coordinates_z']

        #find star particles within 2 kpc of the bh
        centralstar = snapshot.get_particle_data(keys=['Coordinates','Velocities','Masses','Potential'],types=4,center=np.array([ibh_row['Coordinates_x'],ibh_row['Coordinates_y'],ibh_row['Coordinates_z']])*snapshot.units["Coordinates"],radius=2*apy_units.kpc)
        #if no potential data, use the bh location
        if np.isfinite(centralstar['Potential'].values[0]):
            #find the 1000 star particles with the lowest potential energy using a boolean mask
            centralstar.sort_values(by='Potential',ascending=True,inplace=True);centralstar.reset_index(drop=True,inplace=True)
            centralstar = centralstar.iloc[:1000];centralstar.reset_index(drop=True,inplace=True)
            #find the center of potential of these particles weighted by their mass
            poscop = np.average(centralstar.loc[:,['Coordinates_x','Coordinates_y','Coordinates_z']].values,weights=centralstar['Masses'].values,axis=0)
            #find the average velocity of these particles weighted by their mass
            velcop = np.average(centralstar.loc[:,['Velocities_x','Velocities_y','Velocities_z']].values,weights=centralstar['Masses'].values,axis=0)
        else:
            print('No potential data found for star particles. Using BH location as halo minpot.')
            poscop = np.array([ibh_row['Coordinates_x'],ibh_row['Coordinates_y'],ibh_row['Coordinates_z']])
            #select star particles within 2 kpc of the BH
            centralstar = snapshot.get_particle_data(keys=['Coordinates','Velocities','Masses'],types=4,center=poscop,radius=2*apy_units.kpc)
            velcop = np.average(centralstar.loc[:,['Velocities_x','Velocities_y','Velocities_z']].values,weights=centralstar['Masses'].values,axis=0)

        #save the positions and velocities
        halo_output['xminpot'][ibh]=(poscop[0])
        halo_output['yminpot'][ibh]=(poscop[1])
        halo_output['zminpot'][ibh]=(poscop[2])
        halo_output['vx'][ibh]=velcop[0]
        halo_output['vy'][ibh]=velcop[1]
        halo_output['vz'][ibh]=velcop[2]

        #find the overdensity radius and mass
        critdens=delta*cosmo.critical_density(z=snapshot.redshift).to(apy_units.Msun/apy_units.kpc**3).value
        
        #define which center to use
        if useminpot:
            center=apy_units.Quantity([halo_output[f'{x}minpot'][ibh] for x in 'xyz'],unit='kpc')
        else:
            center=apy_units.Quantity([ibh_row['Coordinates_x'],ibh_row['Coordinates_y'],ibh_row['Coordinates_z']],unit='kpc')
        
        if verbose:
            print('Finding virial quantities...')

        #get particle data within 1000 kpc of the center and sort by radius
        pdata_m200=snapshot.get_particle_data(keys=['Coordinates','Masses'],types=[1],center=center,radius=500*apy_units.kpc)
        radius=pdata_m200['R'].values

        sorted_radius=np.argsort(radius)
        sorted_cummass=np.cumsum(pdata_m200['Masses'].values[sorted_radius])
        sorted_radius=radius[sorted_radius];sorted_volume=4/3*np.pi*(sorted_radius)**3
        sorted_cumdens=sorted_cummass/(sorted_volume)

        print(pdata_m200)
        print(sorted_radius)
        print(sorted_cummass)

        iradius=len(sorted_cumdens)-np.searchsorted(sorted_cumdens[::-1],critdens)
    
        #save the mass and radius
        halo_output[f'Halo_M_Crit{delta}'][ibh]=sorted_cummass[iradius]
        halo_output[f'Halo_R_Crit{delta}'][ibh]=sorted_radius[iradius]

    locked_print(f'----> Halo finding complete for {snapshot.snapshot_file.split("/")[-1]} in {time.time()-t0:.2f} seconds.')
    if verbose:
        print()
    
    #return the dataframe
    haloes=pd.DataFrame(halo_output)
    return haloes


# This function is used to find haloes in a snapshot using multiprocessing.
def stack_haloes_worker(snaplist,iproc,delta=200,useminpot=False,verbose=False):

    """
    Find galaxies in several snapshots, for use with multiprocessing.
    
    Parameters
    ----------
    snaplist : list
        List of snapshot objects (or similar) to find haloes in.
    iproc : int
        Index of the process (used for naming the output file).
    delta : float
        Overdensity criteria for the halo finder.  
        Numeric value for overdensity.
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the halo finding.
    
    Returns
    ----------
    None (writes the output to a file).

    """
    #initialize the output
    isnap_outputs=[]

    #for each snapshot in the list, find the haloes
    for snapshot in snaplist:
        isnap_haloes=basic_halofinder(snapshot=snapshot,delta=delta,useminpot=useminpot,verbose=verbose)
        isnap_outputs.append(isnap_haloes)

    #concatenate the outputs
    isnap_haloes=pd.concat(isnap_outputs)
    fname=os.getcwd()+f'/tmphalo/chunk_{str(iproc).zfill(3)}.hdf5'

    #write the output to a file
    if os.path.exists(fname):
        os.remove(fname)
    isnap_haloes.to_hdf(fname,key='chunk')
