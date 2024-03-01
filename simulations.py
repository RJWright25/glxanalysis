
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


# simlations.py
# This file contains the simulation class.

# Import necessary packages
import os
import time
import h5py
import pickle
import numpy as np
import pandas as pd
import astropy.units as apy_units
import astropy.constants as apy_const
import astropy.cosmology as apy_cosmo
import multiprocessing

# Import relevant functions from other files
from .tools import *
from .halofinder import *
from .analysis import *
from .bhinfo import *
from .snapshots import *
from .plotting import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Define the Simulation class (read: a collection of snapshots)
class gadget_simulation:

    """
    A class to represent a collection of GADGET-3 snapshots.

    Parameters:
    -----------
    snapshot_file_list: list
        The list of paths to the snapshot files.


    Attributes:
    -----------
    snapshot_file_list: list
        The list of paths to the snapshot files.
    snapshots: list
        The list of snapshot objects.
    snapshot_idxlist: list
        The list of snapshot indices.
    timelist: list
        The list of times of the snapshots.
    redshiftlist: list
        The list of redshifts of the snapshots.
    hubble: float
        The value of H0/100 from the adopted cosmology.

    """

    # Initialize the simulation object, take a list of snapshot files and create a list of snapshot objects
    def __init__(self, snapshot_file_list, snapshot_type=None):
        if snapshot_type is None:
            snapshot_type = gadget_idealised_snapshot_hki
        elif snapshot_type=='gadget_idealised_snapshot_hki':
            snapshot_type = gadget_idealised_snapshot_hki
        else:
            print('Error: snapshot type not recognized.')
            return None

        self.snapshot_flist = snapshot_file_list;times=[h5py.File(snapshot_file, 'r')['Header'].attrs['Time'] for snapshot_file in self.snapshot_flist]
        self.snapshot_flist = [snapshot_file for _,snapshot_file in sorted(zip(times,self.snapshot_flist))]
        self.snapshots = [snapshot_type(snapshot_file,snapshot_idx=snapshot_idx) for snapshot_idx,snapshot_file in enumerate(self.snapshot_flist)]
        self.snapshot_idxlist = [snapshot.snapshot_idx for snapshot in self.snapshots]
        self.timelist = [snapshot.time for snapshot in self.snapshots]
        self.redshiftlist = [snapshot.redshift for snapshot in self.snapshots]
        self.hubble = self.snapshots[0].hubble
        self.cosmology = self.snapshots[0].cosmology

    # Method to get a snapshot by index or redshift
    def get_snapshot(self, time=None, redshift=None):

        """
        Returns the requested snapshot by index or redshift.

        Parameters:
        -----------
        time: float
            The time of the requested snapshot.
        redshift: float
            The redshift of the requested snapshot.

        Returns:
        -----------
        snapshot : snapshot class
            The requested snapshot.

        """
        #if time is provided, find the snapshot with the closest time
        if time is not None:
            idx = np.argmin(np.abs(np.array(self.timelist)-time))
        #if redshift is provided, find the snapshot with the closest redshift
        elif redshift is not None:
            idx = np.argmin(np.abs(np.array(self.redshiftlist)-redshift))
        else:
            print('Error: must provide either time or redshift')
            return None
        return self.snapshots[idx]
    
    # Method to load the black hole details from a directory
    def load_bhdata(self,path=None,bhids=None,subsample=1):
            
        """
        Load the black hole details from a file.

        Parameters:
        -----------
        path: str
            The path to the directory containing the black hole details files.
        bhids: list
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
            path=self.snapshots[-1].snapshot_file.split('/')[:-1]
            path='/'.join(path)+'/blackhole_details_post_processing/'
        if not os.path.exists(path):
            print('Error: path does not exist.')
            return None        
        
        # use the read_bhdata function from bhprocessing.py to read the black hole details
        bhdata= read_bhdata(self,path=path,bhids=bhids,subsample=subsample)

        # save the black hole details to the simulation object
        self.bhdetails=bhdata

        return bhdata
    
    # Method to find haloes in all snapshots using multiprocessing
    def find_haloes(self,numproc=1,delta=200,useminpot=False,verbose=False):
        
        """
        Find haloes in all snapshots using multiprocessing.

        Parameters:
        -----------
        numproc: int
            The number of processes to use.
        delta: float
            The overdensity criteria for the halo finder.
        useminpot: bool
            If True, use the minimum potential of the star particles as the halo centre.
        verbose: bool
            If True, print the progress of the halo finding.

        Returns:
        -----------
        haloes : pd.DataFrame
            The halo data (see halofinder.py for details).

        """

        print()
        print(f'===========================================================================================')
        print(f'Finding haloes in {len(self.snapshots)} snapshots using {numproc} processes...')
        print(f'===========================================================================================')
        print()

        t0stack=time.time()

        #make a temporary directory for the outputs
        if not os.path.exists(os.getcwd()+'/tmphalo/'):
            os.mkdir(os.getcwd()+'/tmphalo/')
        else:
            for fname in os.listdir(os.getcwd()+'/tmphalo/'):
                if os.path.exists(os.getcwd()+'/tmphalo/'+fname):
                    os.remove(os.getcwd()+'/tmphalo/'+fname)

        #split the snapshots into chunks for multiprocessing
        snapshot_list=self.snapshots
        snapshot_chunks=split_list(snapshot_list,numproc)

        procs=[]
        for iproc in range(numproc):
            snapshots_ichunk=snapshot_chunks[iproc]
            if verbose:
                print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])
            # instantiating process with arguments
            proc = multiprocessing.Process(target=stack_haloes_worker, args=(snapshots_ichunk,iproc,delta,useminpot,verbose))
            procs.append(proc)
            proc.start()

        #complete the processes
        for proc in procs:
            proc.join()
        time.sleep(1)

        #load in outputs and save
        print()
        print('Consolidating halo outputs...')
        chunk_fnames=[os.getcwd()+'/tmphalo/'+file for file in os.listdir(os.getcwd()+'/tmphalo/')]
        chunk_dfs=[pd.read_hdf(fname,key='chunk') for fname in chunk_fnames]
        haloes=pd.concat(chunk_dfs)
        haloes.sort_values(by=['Time','ID'],ascending=[True,True],inplace=True)
        haloes.reset_index(drop=True,inplace=True)

        print()
        print(f'----> Halo finding for {len(self.snapshots)} snaps complete in {time.time()-t0stack:.2f} seconds.')

        self.haloes=haloes
        return haloes
        
    # Method to analyse galaxies in all snapshots using multiprocessing
    def analyse_galaxies(self,numproc=1,shells_kpc=None,useminpot=False,rfac_offset=0.1,verbose=False):
        """
        
        Analyse galaxies in all snapshots using multiprocessing.

        Parameters:
        -----------
        numproc: int
            The number of processes to use.
        shells_kpc: list (of floats, in kpc)
            The radii of the shells to use for the galaxy analysis.
        useminpot: bool
            If True, use the minimum potential of the star particles as the galaxy centre.
        rfac_offset: float
            Fractional value of sphere/shell radius to identify relevant particles.
        verbose: bool
            If True, print the progress of the galaxy analysis.

        Returns:
        -----------
        galaxies : pd.DataFrame
            The galaxy data (see galaxyanalysis.py for details).
        
        """

        print()
        print(f'===========================================================================================')
        print(f'Analysing galaxies in {len(self.snapshots)} snapshots using {numproc} processes...')
        print(f'===========================================================================================')
        print()


        t0stack=time.time()

        #make a temporary directory for the outputs
        if not os.path.exists(os.getcwd()+'/tmpgalx/'):
            os.mkdir(os.getcwd()+'/tmpgalx/')
        else:
            for fname in os.listdir(os.getcwd()+'/tmpgalx/'):
                if os.path.exists(os.getcwd()+'/tmpgalx/'+fname):
                    os.remove(os.getcwd()+'/tmpgalx/'+fname)
        
        #split the snapshots into chunks for multiprocessing
        snapshot_list=self.snapshots
        snapshot_chunks=split_list(snapshot_list,numproc)
        haloes=self.haloes

        #start the processes
        procs=[]
        for iproc in range(numproc):
            snapshots_ichunk=snapshot_chunks[iproc]
            if verbose:
                print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])
            proc = multiprocessing.Process(target=stack_galaxies_worker, args=(snapshots_ichunk,haloes,iproc,shells_kpc,useminpot,rfac_offset,verbose))
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()
        time.sleep(1)

        #load in outputs and save
        print()
        print('Consolidating galaxy outputs...')
        chunk_fnames=[os.getcwd()+'/tmpgalx/'+file for file in os.listdir(os.getcwd()+'/tmpgalx/')]
        chunk_dfs=[pd.read_hdf(fname,key='chunk') for fname in chunk_fnames]
        galaxies=pd.concat(chunk_dfs)
        galaxies.sort_values(by=['Time','ID'],ascending=[True,True],inplace=True)
        galaxies.reset_index(drop=True,inplace=True)

        print()
        print(f'----> Galaxy analysis for {len(self.snapshots)} snaps complete in {time.time()-t0stack:.2f} seconds.')

        self.galaxies=galaxies
        return galaxies
    
    def save_as_pickle(self, fname):
        """
        Save the simulation object as a pickle file.

        Parameters:
        -----------
        fname: str
            The name of the file to save the simulation object to.

        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
    

    ######################################################################################################
    ########################################## Plotting methods ##########################################

    # Method to plot the evolution of the galaxy properties
    def plot_glxevol(self,id=None):
        
        """
        Plot the evolution of the galaxy properties.

        Parameters:
        -----------
        simulation: simulation class
            The simulation object.
        id: int
            The ID of the galaxy to analse.
        
        """
        fig,axes=plot_glxevol(self,id=id)
        return fig,axes

    # Method to plot the evolution pair separation/velocity
    def plot_glxsep(self,id1=None,id2=None):
        """
        Plot the evolution of the pair separation/velocity.
        
        Parameters:
        -----------
        simulation: simulation class
            The simulation object.
        id1: int
            The ID of the first galaxy.
        id2: int
            The ID of the second galaxy.

        """
        fig,axes=plot_glxsep(self,id1=id1,id2=id2)
        return fig,axes
    
    # Method to render all simulation snapshots
    def gen_sim_animation(self,numproc=1,fps=10,type='baryons',frame=None,galaxies=None,useminpot=False,subsample=1,verbose=False):
        """
        Render all simulation snapshots.

        Parameters:
        -----------
        numproc: int
            The number of processes to use.
        fps: int
            The frames per second for the animation.
        type: str
            The type of particles to render.
        frame: float
            The size of the frame to render (in kpc)
        galaxies: pd.DataFrame
            The galaxy data from analyse_galaxies (see galaxyanalysis.py for details).
        useminpot: bool
            If True, use the minimum potential of the star particles as the galaxy centre.
        subsample: int
            The subsampling factor to use when loading the particle data.
        verbose: bool
            If True, print the progress of the rendering.

        """

        #make a directory for the outputs; if it already exists, remove the files
        image_folder=f'plots/render_sim/'
        if not os.path.exists(os.getcwd()+'/plots/'):
            os.mkdir(os.getcwd()+'/plots/')
        if not os.path.exists(os.getcwd()+f'/plots/render_sim/'):
            os.mkdir(os.getcwd()+f'/plots/render_sim/')
        else:
            for fname in os.listdir(os.getcwd()+f'/plots/render_sim/'):
                if os.path.exists(os.getcwd()+f'/plots/render_sim/'+fname):
                    os.remove(os.getcwd()+f'/plots/render_sim/'+fname)
        
        #split the snapshots into chunks for multiprocessing
        snapshot_list=self.snapshots
        snapshots_chunks=split_list(snapshot_list,numproc)

        #start the processes
        procs=[]
        for iproc in range(numproc):
            snapshots_ichunk=snapshots_chunks[iproc]
            if verbose:
                print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])
            proc = multiprocessing.Process(target=render_sim_worker, args=(snapshots_ichunk,type,frame,galaxies,useminpot,subsample,verbose))
            procs.append(proc)
            proc.start()
    
        # complete the processes
        for proc in procs:
            proc.join()
        time.sleep(2)

        #load in snapshots, make a movie
        image_files = sorted([os.path.join(image_folder,img)
                    for img in os.listdir(image_folder)
                    if img.endswith(".png")])
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(f'plots/render_sim/animation_{type}.mp4')
    

    # Method to render a merger 
    def gen_merger_animation(self,numproc=1,fps=10,ids=None,useminpot=False,verbose=False):

        if not np.any(ids):
            haloids_unique=self.galaxies['ID'].unique()[:2]
            ids=sorted([int(haloid) for haloid in haloids_unique])

        image_folder=f'plots/render_merger_{int(ids[0])}_{int(ids[1])}/'
        if not os.path.exists(os.getcwd()+'/plots/'):
            os.mkdir(os.getcwd()+'/plots/')
        if not os.path.exists(os.getcwd()+f'/plots/render_merger_{int(ids[0])}_{int(ids[1])}/'):
            os.mkdir(os.getcwd()+f'/plots/render_merger_{int(ids[0])}_{int(ids[1])}/')
        else:
            for fname in os.listdir(os.getcwd()+f'/plots/render_merger_{int(ids[0])}_{int(ids[1])}/'):
                if os.path.exists(os.getcwd()+f'/plots/render_merger_{int(ids[0])}_{int(ids[1])}/'+fname):
                    os.remove(os.getcwd()+f'/plots/render_merger_{int(ids[0])}_{int(ids[1])}/'+fname)

        snapshot_list=self.snapshots
        snapshots_chunks=split_list(snapshot_list,numproc)

        procs=[]
        for iproc in range(numproc):
            snapshots_ichunk=snapshots_chunks[iproc]
            if verbose:
                print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])            
            proc = multiprocessing.Process(target=render_merger_worker, args=(snapshots_ichunk,self.galaxies,ids,useminpot,verbose))
            procs.append(proc)
            proc.start()
    
        # complete the processes
        for proc in procs:
            proc.join()
        time.sleep(2)

        image_files = sorted([os.path.join(image_folder,img)
                    for img in os.listdir(image_folder)
                    if img.endswith(".png")])
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(f'plots/render_merger_{int(ids[0])}_{int(ids[1])}/animation.mp4')


