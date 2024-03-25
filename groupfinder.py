#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             


# groupfinder.py
# This file contains the function and tools to group galaxies in a snapshot.

import numpy as np
import pandas as pd

# This function is used to group galaxies in the same snapshot.
def basic_groupfinder(galaxies,verbose=False):
    isnaps=galaxies['isnap'].values

    for isnap in np.unique(isnaps):
        if verbose:
            print(f'Grouping galaxies in snapshot {isnap}...')
        
        isnap_mask=galaxies['isnap'].values==isnap

        #initialise the output
        galaxies.loc[isnap_mask,'GroupID']=-1
        galaxies.loc[isnap_mask,'Central']=1
        galaxies.loc[isnap_mask,'RemnantFlag']=0
        galaxies.loc[isnap_mask,'RemnantPartner']=0        
        galaxies.loc[isnap_mask,'RemnantCentral']=-1
        galaxies.loc[isnap_mask,'RemnantSep']=-1
        galaxies.loc[isnap_mask,'CentralDist']=-1

        #galaxy properties
        galaxies_x=galaxies['x'].values
        galaxies_y=galaxies['y'].values
        galaxies_z=galaxies['z'].values
        galaxies_r200=galaxies['Halo_R_Crit200'].values
        galaxies_restar=galaxies['restar_sphere'].values
        galaxies_m200=galaxies['Halo_M_Crit200'].values
        galaxies_mstar=galaxies['1p00restar_sphere_star_tot'].values
        galaxies_IDs=galaxies['ID'].values

        #loop over the galaxies -- find whether any other haloes overlap their R200c
        iigal=-1

        for igal,gal in galaxies.loc[isnap_mask,:].iterrows():
            iigal+=1
            if verbose:
                print(f'Post-processing galaxy {iigal+1}/{galaxies.shape[0]} (ID={int(gal["ID"])})... in snap {isnap}')
            igal_ID=gal['ID']

            #distances
            distances=np.sqrt((galaxies_x-galaxies_x[iigal])**2+(galaxies_y-galaxies_y[iigal])**2+(galaxies_z-galaxies_z[iigal])**2)
            m200_offsets=np.abs(np.log10(galaxies_m200/galaxies_m200[iigal]))
            mstar_offsets=np.abs(np.log10(galaxies_mstar/galaxies_mstar[iigal]))
            
            #### GROUP FINDING ####
            #find the potential group members as galaxies within 2R200c
            group_mask=np.logical_and(isnap_mask,distances<1*galaxies_r200[iigal])
            #iteratively find the galaxies within 2R200c of the largest R200c until no more are found
            numgals_thisgroup=np.nansum(group_mask)
            for iiter in range(100): 
                #find the largest R200c in the group
                r200_to_use=np.max(galaxies_r200[group_mask])
                #find the galaxies within 2R200c of the largest R200c
                group_mask=np.logical_and(isnap_mask,np.logical_or(group_mask,distances<1*r200_to_use))
                #check if any new galaxies are found
                group_ids_thisgroup=galaxies['GroupID'].values[isnap_mask][group_mask]
                for group_id_thisgroup in group_ids_thisgroup[group_ids_thisgroup>0]:
                    group_mask=np.logical_and(isnap_mask,np.logical_or(group_mask,galaxies['GroupID'].values[isnap_mask]==group_id_thisgroup))
                #check if the number of galaxies in the group has changed
                if np.nansum(group_mask)>numgals_thisgroup:
                    numgals_thisgroup=np.nansum(group_mask)
                else:
                    break
            

            #check if any of the galaxies are already in a group
            group_ids_thisgroup=galaxies['GroupID'].values[group_mask]
            if np.nansum(group_ids_thisgroup>0):
                group_id=group_ids_thisgroup[group_ids_thisgroup>0][0]
            else:
                group_id=isnap*1e4+iigal
            
            #assign group ID and find central galaxy
            if np.nansum(group_mask):
                galaxies.loc[group_mask,'GroupID']=group_id
                galaxies.loc[group_mask,'Central']=0

                #central is the galaxy with the largest BH mass
                igals_thisgroup_mstar=galaxies['1p00restar_sphere_star_tot'].values[group_mask]
                igals_thisgroup_mstar_largest_ID=galaxies['ID'].values[group_mask][np.argmax(igals_thisgroup_mstar)]
                galaxies.loc[np.logical_and(isnap_mask,igals_thisgroup_mstar_largest_ID==galaxies['ID'].values),'Central']=1

            #### PARTNER FINDING ####
            #check if already a remnant
            if galaxies.loc[igal,'RemnantFlag']==1:
                continue
            else:
                #find the potential remnant partners as galaxies within 4*Restar and within 0.1 dex in Mstar and M200c
                partner_mask=np.logical_and(distances<4*galaxies_restar[iigal],distances>=1e-4)
                partner_mask=np.logical_and(partner_mask,mstar_offsets<0.1)
                partner_mask=np.logical_and(partner_mask,m200_offsets<0.1)
                partner_mask=np.logical_and(partner_mask,isnap_mask)

                if np.nansum(partner_mask):
                    galaxies.loc[partner_mask,'RemnantFlag']=1;galaxies.loc[igal,'RemnantFlag']=1
                    galaxies.loc[partner_mask,'RemnantPartner']=igal_ID;galaxies.loc[igal,'RemnantPartner']=galaxies_IDs[partner_mask][0]
                    galaxies.loc[partner_mask,'RemnantCentral']=0;galaxies.loc[igal,'RemnantPartner']=0

                    #find central as the galaxy with the largest BH mass
                    igal_bhmass=gal['BH_Mass'];igal_partner_bhmass=galaxies['BH_Mass'].values[partner_mask][0]
                    if igal_bhmass>igal_partner_bhmass:
                        galaxies.loc[igal,'RemnantCentral']=1
                    else:
                        galaxies.loc[partner_mask,'RemnantCentral']=1
                    #remnant sep
                    galaxies.loc[igal,'RemnantSep']=distances[partner_mask][0]
                    galaxies.loc[partner_mask,'RemnantSep']=distances[partner_mask][0]

        #get halo-centric distance for all satellites
        group_ids=np.unique(galaxies['GroupID'].values[isnap_mask])
        for group_id in group_ids:
            if group_id>0:
                group_mask=np.logical_and(galaxies['GroupID'].values==group_id,isnap_mask)
                if np.nansum(group_mask):
                    group_central_mask=np.logical_and(group_mask,galaxies['Central'].values==1)
                    group_xyz=np.array([galaxies.loc[group_central_mask,f'{x}'].values[0] for x in 'xyz'])
                    galaxies_xyz=np.array([galaxies['x'].values[group_mask],galaxies['y'].values[group_mask],galaxies['z'].values[group_mask]]).T
                    galaxies.loc[group_mask,'CentralDist']=np.sqrt(np.sum((galaxies_xyz-group_xyz)**2,axis=1))

    return galaxies
            