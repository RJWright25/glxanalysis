
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
        isnap_igals=np.where(isnap_mask)[0]

        #initialise the output
        galaxies.loc[isnap_mask,'GroupID']=-1
        galaxies.loc[isnap_mask,'Central']=1
        galaxies.loc[isnap_mask,'RemnantFlag']=0
        galaxies.loc[isnap_mask,'RemnantPartner']=0        
        galaxies.loc[isnap_mask,'RemnantCentral']=-1
        galaxies.loc[isnap_mask,'RemnantSep']=-1
        galaxies.loc[isnap_mask,'CentralDist']=-1

        #galaxy properties
        galaxies_x=galaxies['x'].values[isnap_mask]
        galaxies_y=galaxies['y'].values[isnap_mask]
        galaxies_z=galaxies['z'].values[isnap_mask]
        galaxies_r200=galaxies['Halo_R_Crit200'].values[isnap_mask]
        galaxies_restar=galaxies['restar_sphere'].values[isnap_mask]
        galaxies_m200=galaxies['Halo_M_Crit200'].values[isnap_mask]
        galaxies_mstar=galaxies['1p00restar_sphere_star_tot'].values[isnap_mask]
        galaxies_IDs=galaxies['ID'].values[isnap_mask]

        #loop over the galaxies -- find whether any other haloes overlap their R200c
        iigal=-1

        for igal,gal in galaxies.loc[isnap_mask,:].iterrows():
            iigal+=1
            if verbose:
                print(f'Post-processing galaxy {igal+1}/{galaxies.shape[0]} (ID={int(gal["ID"])})... in snap {isnap}')
            igal_ID=gal['ID']

            #distances
            distances=np.sqrt((galaxies_x-galaxies_x[iigal])**2+(galaxies_y-galaxies_y[iigal])**2+(galaxies_z-galaxies_z[iigal])**2)
            m200_offsets=np.abs(np.log10(galaxies_m200/galaxies_m200[iigal]))
            mstar_offsets=np.abs(np.log10(galaxies_mstar/galaxies_mstar[iigal]))
            
            #### GROUP FINDING ####
            #find the potential group members as galaxies within 2R200c
            group_mask=distances<2*galaxies_r200[iigal]
            #iteratively find the galaxies within 2R200c of the largest R200c until no more are found
            numgals_thisgroup=np.nansum(group_mask)
            for iiter in range(100): 
                #find the largest R200c in the group
                r200_to_use=np.max(galaxies_r200[group_mask])
                #find the galaxies within 2R200c of the largest R200c
                group_mask=np.logical_or(group_mask,distances<2*r200_to_use)
                #check if any new galaxies are found
                group_ids_thisgroup=galaxies['GroupID'].values[isnap_mask][group_mask]
                for group_id_thisgroup in group_ids_thisgroup[group_ids_thisgroup>0]:
                    group_mask=np.logical_or(group_mask,galaxies['GroupID'].values[isnap_mask]==group_id_thisgroup)
                #check if the number of galaxies in the group has changed
                if np.nansum(group_mask)>numgals_thisgroup:
                    numgals_thisgroup=np.nansum(group_mask)
                else:
                    break
            
            #check if any of the galaxies are already in a group
            group_ids_thisgroup=galaxies['GroupID'].values[isnap_mask][group_mask]
            if np.nansum(group_ids_thisgroup>0):
                group_id=group_ids_thisgroup[group_ids_thisgroup>0][0]
            else:
                group_id=isnap*1e4+iigal

            #assign group ID and find central galaxy
            if np.nansum(group_mask):
                igals_thisgroup=isnap_igals[group_mask]
                for igal_igroup in igals_thisgroup:
                    galaxies.loc[igal_igroup,'GroupID']=group_id
                    galaxies.loc[igal_igroup,'Central']=0
                #central is the galaxy with the largest BH mass
                igals_thisgroup_bhmass=galaxies['BH_Mass'].values[isnap_mask][group_mask]
                galaxies.loc[igals_thisgroup[np.argmax(igals_thisgroup_bhmass)],'Central']=1

            #### PARTNER FINDING ####
            #check if already a remnant
            if galaxies.loc[igal,'RemnantFlag']==1:
                continue
            else:
                #find the potential remnant partners as galaxies within 4*Restar and within 0.1 dex in Mstar and M200c
                partner_mask=np.logical_and(distances<4*galaxies_restar[iigal],distances>=1e-4)
                partner_mask=np.logical_and(partner_mask,mstar_offsets<0.1)
                partner_mask=np.logical_and(partner_mask,m200_offsets<0.1)

                if np.nansum(partner_mask):
                    igal_partner=isnap_igals[partner_mask][0]
                    galaxies.loc[igal,'RemnantFlag']=1;galaxies.loc[igal_partner,'RemnantFlag']=1
                    galaxies.loc[igal,'RemnantPartner']=galaxies_IDs[partner_mask][0];galaxies.loc[igal_partner,'RemnantPartner']=igal_ID
                    galaxies.loc[igal,'RemnantCentral']=0;galaxies.loc[igal_partner,'RemnantCentral']=0

                    #find central as the galaxy with the largest BH mass
                    igal_bhmass=gal['BH_Mass'];igal_partner_bhmass=galaxies['BH_Mass'].values[isnap_mask][partner_mask][0]
                    if igal_bhmass>igal_partner_bhmass:
                        galaxies.loc[igal,'RemnantCentral']=1
                    else:
                        galaxies.loc[igal_partner,'RemnantCentral']=1
                    #remnant sep
                    galaxies.loc[igal,'RemnantSep']=distances[partner_mask][0]
                    galaxies.loc[igal_partner,'RemnantSep']=distances[partner_mask][0]

        #get halo-centric distance for all satellites
        group_ids=np.unique(galaxies['GroupID'].values[isnap_mask])
        for group_id in group_ids:
            if group_id>0:
                group_mask=galaxies.loc[isnap_mask,'GroupID'].values==group_id
                if np.nansum(group_mask):
                    group_igals=np.where(isnap_mask)[0][group_mask]
                    group_central_mask=np.logical_and(group_mask,galaxies.loc[isnap_mask,'Central'].values==1)
                    group_xyz=np.array([galaxies.loc[isnap_mask,f'{x}'].values[group_central_mask][0] for x in 'xyz'])
                    galaxies_xyz=np.array([galaxies['x'].values[isnap_mask][group_mask],galaxies['y'].values[isnap_mask][group_mask],galaxies['z'].values[isnap_mask][group_mask]]).T
                    distances=np.sqrt(np.sum((galaxies_xyz-group_xyz)**2,axis=1))
                    for group_igal,dist in zip(group_igals,distances):
                        galaxies.loc[group_igal,'CentralDist']=dist


    return galaxies
            