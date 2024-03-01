
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


# plotting.py
# This file contains a few functions to plot key results from the code.

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sphviewer
import moviepy

from .tools import locked_print

# Default matplotlib settings and color selections
plt.style.use('https://raw.githubusercontent.com/RJWright25/analysis/master/mplparams.txt')
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
dpi=500

# Define cmaps for gas and stars
cname_star='#FDF9E8'
cmapname_gas='magma'
cmap_gas = plt.get_cmap(cmapname_gas, 256)
cmaplist_gas = cmap_gas(np.linspace(0, 1, 256))
for ival,cmapval in enumerate(cmaplist_gas):
    hsv=matplotlib.colors.rgb_to_hsv(cmapval[:3])
    cmaplist_gas[ival,:3] = matplotlib.colors.hsv_to_rgb(hsv)
    cmaplist_gas[ival,-1] = (ival+1)/256
cmap_gas = matplotlib.colors.ListedColormap(cmaplist_gas)

# This function is used to plot the evolution of the properties of a galaxy specified by its ID.
def plot_glxevol(simulation,id=None):
    haloes=simulation.haloes
    galaxies=simulation.galaxies
    bhdetails=simulation.bhdetails

    #if no id is given, take the first halo
    if not id:
        #figure out which halo is the primary
        ids=haloes['ID'].unique()
        ids_shape={id:galaxies.loc[galaxies['ID'].values==id,:].shape[0] for id in ids}
        id=[id for id in ids if ids_shape[id]==np.max(list(ids_shape.values()))][0]

    galaxy_masked=galaxies.loc[galaxies['ID'].values==id,:]
    bhdetails_masked=bhdetails[id]

    #kernels
    snapkernel=np.ones(1)/1;bhkernel=np.ones(10)/10

    #times
    snaptime=np.convolve(galaxy_masked['Time'].values,snapkernel,mode='valid')
    bhtime=np.convolve(bhdetails_masked['Time'].values,bhkernel,mode='valid')

    #mass
    mass_star=np.convolve(galaxy_masked['2p00restar_sphere_star_tot'].values,snapkernel,mode='valid')
    mass_bh=np.convolve(bhdetails_masked['bh_M'].values,bhkernel,mode='valid')

    #sfr and outflow/inflow
    sfr=np.convolve(galaxy_masked['2p00restar_sphere_gas_sfr'].values,snapkernel,mode='valid')
    inflow=np.convolve(galaxy_masked['2p00restar_shell_gasinflow_all_mdot'].values,snapkernel,mode='valid')
    outflow=np.convolve(galaxy_masked['2p00restar_shell_gasoutflow_all_mdot'].values,snapkernel,mode='valid')
    
    #figure
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(6,2.5),gridspec_kw={'left':0.15,'right':0.95,'bottom':0.1,'top':0.95,'hspace':0.2,'wspace':0.3})
    fig.set_dpi(dpi)
    for ax in axes:
        ax.grid(True,which='major',alpha=1)

    #mass
    axes[0].plot(snaptime,mass_star,c='k',lw=2.5,alpha=0.75)
    axes[0].plot(snaptime,mass_star,c='goldenrod',lw=1.5, label=r'$M_{\star}$ $(2\times R_{\rm eff})$')
    axes[0].plot(bhtime,mass_bh,c='grey',lw=1,alpha=0.5, label=r'$M_{\rm BH}$'+f' ({int(id)})')

    axes[0].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[0].set_xlim(snaptime[0],snaptime[-1])
    axes[0].set_ylabel(r'$M\, [{\rm M}_{\odot}]$')
    axes[0].set_yscale('log')
    axes[0].legend(loc='lower center')

    #sfr
    axes[1].plot(snaptime,sfr,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,sfr,c='grey',lw=1.5, label=r'SFR $(2\times R_{\rm eff})$')

    axes[1].plot(snaptime,inflow,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,inflow,c='C0',lw=1.5, label=r'$\dot{M}_{\rm in} \, (2\times R_{\rm eff})$')

    axes[1].plot(snaptime,outflow,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,outflow,c='C1',lw=1.5, label=r'$\dot{M}_{\rm out} \, (2\times R_{\rm eff})$')


    axes[1].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[1].set_xlim(snaptime[0],snaptime[-1])
    axes[1].set_ylabel(r'$\dot{M}$ [${\rm M}_{\odot}\,{\rm yr}^{-1}$]')
    axes[1].set_yscale('log')
    axes[1].legend(loc='lower center')

    if not os.path.exists(os.getcwd()+'/plots/'):
        os.mkdir(os.getcwd()+'/plots/')

    fig.set_dpi(dpi)
    plt.savefig(os.getcwd()+'/plots/'+f'glxevol_{int(id)}.png',bbox_inches='tight')

    return fig,axes


# This function is used to plot the separation and relative velocity of two galaxies specified by their IDs.
def plot_glxsep(simulation,id1=None,id2=None):
    
    """
    Plots the separation and relative velocity of two galaxies specified by their IDs.

    Parameters
    ----------
    simulation : simulation object
        Simulation object containing the data to be plotted.
        NB: The simulation object must contain the following data:
        - haloes: pandas dataframe containing the properties of the halos found in the snapshot.
        - galaxies: pandas dataframe containing the properties of the galaxies found in the snapshot.
        - bhdetails: dictionary of pandas dataframes containing the properties of the black holes found in the snapshot.
    id1 : int
        ID (BH) of the first galaxy.
    id2 : int
        ID (BH) of the second galaxy.

    Returns
    ----------
    fig : matplotlib figure
        The figure object containing the plot.
    axes : matplotlib axes
        The axes object containing the plot.

    """

    haloes=simulation.haloes
    galaxies=simulation.galaxies
    bhdetails=simulation.bhdetails

    #if no ids are given, take the first two haloes
    if not id1 or not id2:
        haloids=haloes['ID'].unique()[:2]
        id1=haloids[0];id2=haloids[1]
    else:
        haloids=[id1,id2]
    
    #mask galaxies and bhdetails
    galaxies_masked={id:galaxies.loc[galaxies['ID'].values==id,:] for id in haloids}
    bhdetails_masked={id:bhdetails[id] for id in haloids}

    #times
    idx_merger=np.nanmin([galaxies_masked[id].shape[0] for id in haloids])
    snaptime=galaxies_masked[id1]['Time'].values[:idx_merger]

    #separation
    xsep=galaxies_masked[haloids[0]]['xminpot'].values[:idx_merger]-galaxies_masked[haloids[1]]['xminpot'].values[:idx_merger]
    ysep=galaxies_masked[haloids[0]]['yminpot'].values[:idx_merger]-galaxies_masked[haloids[1]]['yminpot'].values[:idx_merger]
    zsep=galaxies_masked[haloids[0]]['zminpot'].values[:idx_merger]-galaxies_masked[haloids[1]]['zminpot'].values[:idx_merger]
    sep=np.sqrt(xsep**2+ysep**2+zsep**2)

    #relative velocity
    vxsep=galaxies_masked[haloids[0]]['vx'].values[:idx_merger]-galaxies_masked[haloids[1]]['vx'].values[:idx_merger]
    vysep=galaxies_masked[haloids[0]]['vy'].values[:idx_merger]-galaxies_masked[haloids[1]]['vy'].values[:idx_merger]
    vzsep=galaxies_masked[haloids[0]]['vz'].values[:idx_merger]-galaxies_masked[haloids[1]]['vz'].values[:idx_merger]
    vrel=np.sqrt(vxsep**2+vysep**2+vzsep**2)

    #bh separation
    #which bh lives the longest
    bhids=[int(bh) for bh in bhdetails_masked.keys()]
    bhids_shape={bh:bhdetails_masked[bh].shape[0] for bh in bhids}
    bhid_remnant=[bh for bh in bhids if bhids_shape[bh]==np.max(list(bhids_shape.values()))][0]
    bhid_sec=[bh for bh in bhids if bh!=bhid_remnant][0]

    #match time-step from secondary to primary
    time_sec=bhdetails_masked[bhid_sec]['Time'].values
    time_rem=bhdetails_masked[bhid_remnant]['Time'].values
    sep_bh=np.zeros(time_sec.shape[0])
    vel_bh=np.zeros(time_sec.shape[0])
    
    #for each time in the secondary, find the idx of the closest time in the primary and get sep/vel at that idx
    for itime,time in enumerate(time_sec):
        idx_prim=np.argmin(np.abs(time_rem-time))
        xyz_sec=bhdetails_masked[bhid_sec].loc[:,['Coordinates_x','Coordinates_y','Coordinates_z']].values[itime,:]
        xyz_rem=bhdetails_masked[bhid_remnant].loc[:,['Coordinates_x','Coordinates_y','Coordinates_z']].values[idx_prim,:]
        vel_sec=bhdetails_masked[bhid_sec].loc[:,['V_x','V_y','V_z']].values[itime,:]
        vel_rem=bhdetails_masked[bhid_remnant].loc[:,['V_x','V_y','V_z']].values[idx_prim,:]
        vel_bh[itime]=np.sqrt(np.sum((vel_sec-vel_rem)**2))
        sep_bh[itime]=np.sqrt(np.sum((xyz_sec-xyz_rem)**2))

    #r200 and restar
    r200_0=galaxies_masked[haloids[0]]['Halo_R_Crit200'].values[:idx_merger]
    # r200_1=galaxies_masked[haloids[1]]['Halo_R_Crit200'].values[:idx_merger]

    #find tlims
    tlims=(snaptime[0]-0.1,snaptime[-1]+0.1)

    #figure
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(6,2.5))
    fig.set_dpi(dpi)
    for ax in axes:
        ax.grid(True,which='major',alpha=1)

    #separation
    axes[0].plot(snaptime,sep,c='k',lw=2.5)
    axes[0].plot(snaptime,sep,c='grey',lw=1.5, label=r'Halo separation')
    
    axes[0].plot(time_sec,sep_bh,c='grey',lw=1,alpha=0.5,label=r'BH separation')

    axes[0].plot(snaptime,r200_0,c='k',lw=2.5)
    axes[0].plot(snaptime,r200_0,c='maroon',lw=1.5,label=r'Primary $R_{\rm 200c}$')
    axes[0].legend(loc='lower center')

    axes[0].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[0].set_xlim(tlims)
    axes[0].set_ylabel(r'Separation [kpc]')
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.1,1e3)

    #relative vel
    smoothn=1;kernel=np.ones(smoothn)/smoothn
    axes[1].plot(np.convolve(snaptime,kernel,mode='valid'),np.convolve(vrel,kernel,mode='valid'),c='k',lw=2.5)
    axes[1].plot(np.convolve(snaptime,kernel,mode='valid'),np.convolve(vrel,kernel,mode='valid'),c='grey',lw=1.5, label=r'Halo $v_{\rm rel}$')

    axes[1].plot(time_sec,vel_bh,c='grey',lw=1,alpha=0.5,label=r'BH $v_{\rm rel}$')

    axes[1].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[1].set_xlim(tlims)
    axes[1].set_ylabel(r'Relative velocity [${\rm km}\,{\rm s}^{-1}$]')
    axes[1].set_yscale('log')
    axes[1].legend(loc='lower center')

    if not os.path.exists(os.getcwd()+'/plots/'):
        os.mkdir(os.getcwd()+'/plots/')

    fig.set_dpi(dpi)
    plt.savefig(os.getcwd()+'/plots/'+f'glxsep_{int(id1)}_{int(id2)}.png',bbox_inches='tight')

    return fig,axes

def render_snap(snapshot,type='baryons',frame=None,galaxies=None,useminpot=False,subsample=1,verbose=False):
    """
    Render a snapshot of the simulation.

    Parameters
    ----------
    snapshot : snapshot object
        Snapshot object to render.
    type : str
        Type of rendering to perform. Options are 'baryons' and 'dm'.
    galaxies : pandas dataframe
        Dataframe containing the properties of the galaxies found in the snapshots (optional).
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the rendering.
    """

    if type=='baryons':
        ptypes=[0,4];radstr='restar_sphere'
        cmap=cmap_gas
    elif type=='dm':
        ptypes=[1];radstr='Halo_R_Crit200'
        cmap='viridis'
    else:
        print('Type not recognized. Options are "baryons" and "dm".')
        return

    if not frame:
        frame=snapshot.boxsize/2

    censtr=''
    if useminpot:censtr='minpot'

    pdata=snapshot.get_particle_data(keys=['Coordinates','Masses'], types=ptypes, center=None, radius=None,subsample=subsample)
    sph_fluidmask=pdata['ParticleTypes'].values==ptypes[0]
    sph_particles=sphviewer.Particles(pdata.loc[sph_fluidmask,[f'Coordinates_{x}' for x in 'xyz']].values,
                                      pdata.loc[sph_fluidmask,'Masses'].values,nb=8)
    sph_camera = sphviewer.Camera(r='infinity', t=0, p=0, roll=0, xsize=1500, ysize=1500,
                                                x=0, y=0, z=0,
                                                extent=[-frame,frame,-frame,frame])
    sph_scene = sphviewer.Scene(sph_particles,sph_camera)
    sph_render = sphviewer.Render(sph_scene)
    sph_extent = sph_render.get_extent()
    sph_img=sph_render.get_image()
    
    #make figure and plot
    fig,ax=plt.subplots(1,1,figsize=(5,5),gridspec_kw={'left':0.1,'right':0.99,'bottom':0.1,'top':0.99})
    ax.set_facecolor('k')
    ax.grid(which='both',alpha=0)
    ax.imshow(sph_img,extent=sph_extent,origin='lower',cmap=cmap,norm=matplotlib.colors.LogNorm(),zorder=1)
    
    #add stars if necessary
    if type=='baryons':
        stars=pdata.loc[pdata['ParticleTypes'].values==4,:]
        ax.scatter(stars.loc[:,'Coordinates_x'].values,stars.loc[:,'Coordinates_y'].values,c=cname_star,alpha=0.03,s=0.05,lw=0,zorder=2)

    #add galaxy positions
    relevant_galaxies=galaxies.loc[galaxies['isnap'].values==snapshot.snapshot_idx,:]
    if relevant_galaxies.shape[0]:
        for igal,gal in relevant_galaxies.iterrows():
            ax.scatter(gal[f'x{censtr}'],gal[f'y{censtr}'],s=2,c='w',zorder=2)
            ax.scatter(gal[f'x{censtr}'],gal[f'y{censtr}'],s=1,c='k',zorder=2)
            ax.add_artist(plt.Circle(radius=gal[radstr],xy=[gal[f'x{censtr}'],gal[f'y{censtr}']],color='w',lw=0.5,ls='--',fill=False,zorder=2))
    
    ax.set_xlim(-frame,frame)
    ax.set_ylim(-frame,frame)

    ax.text(0.55,0.01,'$x$ [kpc]',transform=fig.transFigure,ha='center',va='bottom')
    ax.text(0.01,0.55,'$y$ [kpc]',transform=fig.transFigure,ha='left',va='center',rotation=90)
    ax.text(x=0.95,y=0.95,s=r'$t='+f'{snapshot.time:.3f}$ Gyr',transform=ax.transAxes,ha='right',va='top',color='w')
    
    fig.set_dpi(dpi)

    return fig,ax


def render_sim_worker(snaplist,type='baryons',frame=None,galaxies=None,useminpot=False,subsample=1,verbose=False):
    
    """
    Worker function to make an animation of the simulation for a given set of snaps.

    Parameters
    ----------
    snaplist : list
        List of snapshot objects (or similar) to use in the animation.
    type : str
        Type of rendering to perform. Options are 'baryons' and 'dm'.
    frame : float
        Size of the frame to use in the rendering.
    galaxies : pandas dataframe
        Dataframe containing the properties of the galaxies found in the snapshots (optional).
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the animation.
    
    """
    
    for snapshot in snaplist:
        if verbose:
            locked_print(f"Rendering snap {snapshot.snapshot_idx}...")
        fig,_=render_snap(snapshot,type=type,frame=frame,galaxies=galaxies,useminpot=useminpot,subsample=subsample,verbose=verbose)
        fig.savefig(f'plots/render_sim/snap_{str(snapshot.snapshot_idx).zfill(3)}.png',bbox_inches='tight')
        plt.close(fig)
    

# This function is used to create an animation of the interaction between two galaxies specified by their IDs.
def render_merger_worker(snaplist,galaxies,ids=None,useminpot=False,verbose=False):
    
    """
    Worker function to make an animation of the interaction between two galaxies specified by their IDs.

    Parameters
    ----------
    snaplist : list
        List of snapshot objects (or similar) to use in the animation.
    galaxies : pandas dataframe
        Dataframe containing the properties of the galaxies found in the snapshots.
    ids : list
        List of galaxy IDs to use in the animation.
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the animation.

    Returns
    ----------
    None (writes the output to a file).
    
    """

    #make an animation of stars & gas
    if not ids:
        haloids_unique=galaxies['ID'].unique()[::-1][:2]
    else:
        haloids_unique=ids

    #id remaining
    remnantid=haloids_unique[np.nanargmax([galaxies.loc[galaxies['ID'].values==haloid,:].shape[0] for haloid in haloids_unique])]
    secondid=haloids_unique[np.nanargmin([galaxies.loc[galaxies['ID'].values==haloid,:].shape[0] for haloid in haloids_unique])]

    for snapshot in snaplist:
        isnap=snapshot.snapshot_idx
        isnap_gals=galaxies.loc[galaxies['isnap'].values==isnap,:]
        isnap_primary=isnap_gals.loc[isnap_gals['ID'].values==remnantid,:];isnap_primary.reset_index(drop=True,inplace=True)
        isnap_secondary=isnap_gals.loc[isnap_gals['ID'].values==secondid,:];isnap_secondary.reset_index(drop=True,inplace=True)
        num_gals=isnap_gals.shape[0]
        if num_gals<2:
            merged=True
        elif num_gals==2:
            merged=False
        
        #get xysep
        poskey=''
        if useminpot:
            poskey='minpot'
        if not merged:
            x1=isnap_primary[f'x{poskey}'].values[0];x2=isnap_secondary[f'x{poskey}'].values[0]
            y1=isnap_primary[f'y{poskey}'].values[0];y2=isnap_secondary[f'y{poskey}'].values[0]
            z1=isnap_primary[f'z{poskey}'].values[0];z2=isnap_secondary[f'z{poskey}'].values[0]
            rad1=isnap_primary['restar_sphere'].values[0];rad2=isnap_secondary['restar_sphere'].values[0]
            xysep=np.sqrt((x1-x2)**2+(y1-y2)**2)
            center=np.array([(x1+x2)/2,(y1+y2)/2,(z1+z2)/2])
        else:
            x1=isnap_primary[f'x{poskey}'].values[0]
            y1=isnap_primary[f'y{poskey}'].values[0]
            z1=isnap_primary[f'z{poskey}'].values[0]
            x2=np.nan;y2=np.nan;z2=np.nan
            rad1=isnap_gals['restar_sphere'].values[0];rad2=np.nan
            xysep=0
            center=np.array([x1,y1,z1])

        fig,ax=plt.subplots(1,1,figsize=(5,5),gridspec_kw={'left':0.1,'right':0.99,'bottom':0.1,'top':0.99})
        ax.set_facecolor('k')
        ax.grid(which='both',alpha=0)

        if verbose:
            locked_print(f"Rendering snap {isnap}...")

        frame=np.nanmax([xysep*1,25])

        pdata=snapshot.get_particle_data(keys=['Coordinates','Masses'], types=[0,4], center=None, radius=None,subsample=1)
        stars=pdata.loc[pdata['ParticleTypes'].values==4,:]
        gas=pdata.loc[pdata['ParticleTypes'].values==0,:]

        #sph rendering
        sph_particles=sphviewer.Particles(gas.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-center,gas['Masses'].values,nb=8)
        sph_camera = sphviewer.Camera(r='infinity', t=0, p=0, roll=0, xsize=1500, ysize=1500,
                                                    x=0, y=0, z=0,
                                                    extent=[-frame,frame,-frame,frame])
        sph_scene=sphviewer.Scene(sph_particles,sph_camera)
        sph_render = sphviewer.Render(sph_scene)
        sph_extent = sph_render.get_extent()
        sph_img=sph_render.get_image()
        sph_img[sph_img==0]

        # ax.fill_between([-200,200],[-200,-200],[200,200],color='k',alpha=1,zorder=0)
        ax.imshow(sph_img,extent=sph_extent,origin='lower',cmap=cmap_gas,norm=matplotlib.colors.LogNorm(1e4,1e9),zorder=1)
        ax.scatter(stars.loc[:,'Coordinates_x'].values-center[0],stars.loc[:,'Coordinates_y'].values-center[1],c=cname_star,alpha=0.03,s=0.05,lw=0,zorder=2)

        #plot the galaxies
        ax.scatter(x1-center[0],y1-center[1],s=2,c=f'w',zorder=2)
        ax.scatter(x1-center[0],y1-center[1],s=1,c=f'k',zorder=2)
        ax.add_artist(plt.Circle(radius=rad1,xy=[x1-center[0],y1-center[1]],color=f'w',lw=0.5,ls='--',fill=False,zorder=2))
        
        if not merged:
            ax.scatter(x2-center[0],y2-center[1],s=2,c=f'w',zorder=2)
            ax.scatter(x2-center[0],y2-center[1],s=1,c=f'k',zorder=2)
            ax.add_artist(plt.Circle(radius=rad2,xy=[x2-center[0],y2-center[1]],color=f'w',lw=0.5,ls='--',fill=False,zorder=2))
        else:
            ax.text(x=0.05,y=0.95,s='Merged',transform=ax.transAxes,ha='left',va='top',color='w')
        
        ax.set_xlim(-frame,frame)
        ax.set_ylim(-frame,frame)
        ax.text(0.55,0.01,'$x$ [kpc]',transform=fig.transFigure,ha='center',va='bottom')
        ax.text(0.01,0.55,'$y$ [kpc]',transform=fig.transFigure,ha='left',va='center',rotation=90)
        ax.text(x=0.95,y=0.95,s=r'$t='+f'{snapshot.time:.3f}$ Gyr',transform=ax.transAxes,ha='right',va='top',color='w')
        
        plt.savefig(f'plots/render_merger_{int(ids[0])}_{int(ids[1])}/snap_{str(isnap).zfill(3)}.png',dpi=dpi)
        plt.close()


