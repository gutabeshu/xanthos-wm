"""
Natural System Components
@date   10/14/2016
@author: lixi729
@email: xinya.li@pnl.gov
@Project: Xanthos V1.0

License:  BSD 2-Clause, see LICENSE and DISCLAIMER files
Copyright (c) 2017, Battelle Memorial Institute
'''
'''
Water Management Components
@date   08/22/2021
@authors: Guta Wakbulcho Abeshu : gwabeshu@uh.edu
          University of Houston   
          HongYi Li : hli57@uh.edu
          University of Houston                
"""

import os
import math
import scipy.io as scio
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from datetime import date


def streamrouting(L, S0, F0, ChV, q, area, nday, dt, UM, UP, Sini_byr, wdirr, irrmean, 
                  mtifl, ppose, cpa,Release_policy, maxTurbineFlow, WConsumption, 
				  alpha, Sini_resv, res_flag, grdc_us_grids):
    """
    Runoff routing with water management

    L:    flow distance (m)                                        = (N x 1)
    S0:   initial channel storage value for the month (m^3)        = (N x 1)
    F0:   initial channel flow value (instantaneous) (m^3/s)       = (N x 1)
    ChV:  channel velocity (m/s)                                   = (N x 1)
    q:    runoff (mm/month)                                        = (N x 1)
    area: cell area (km^2)                                         = (N x 1)
    nday: number of days in the month                              = (1 x 1)
    dt:   size of the fixed time step (s)                          = (1 x 1)
    UM:   connection matrix (see notes in upstream_genmatrix)      = (N x N, sparse)
    Release_policy: policy look up table for Hydropower release
    maxTurbineFlow: maximum turbine flow
    WConsumption: water consumption
    alpha : reservoir capacity reduction coeffitient
    Sini_byr : initial reservoir storage at begining of the year
    Sini_resv: reservoir storage at end of preceeding month
    wdirr: irrigation demand
    irrmean: mean irrigation demand
    mtifl : mean total inflow
    ppose : reservoir purpose
    cpa: reservoir capacity
    res_flag: 0 for no water management and 1 for water management

    Outputs:
    S:     channel storage, unit m3
    Favg:  monthly average channel flow, unit m3/s
    F:     instantaneous channel flow, unit m3/s
    Sending : reservoir storage at end of month, unit m3
    Qin_res_avg: reservoir inflow, unit m3/s
    Qout_res_avg : reservoir outflow, unit m3/s
    """

    N = L.shape[0]                   # number of cells
    nt = int(nday * 24 * 3600 / dt)  # number of time steps

    # Initialite
    S = np.copy(S0) 
    F = np.copy(F0)
    Favg = np.zeros((N,), dtype='f8')            # Mean Channel flow
    Qin_res_avg = np.zeros((N,), dtype='f8')     # Inflow to reservoir
    Qout_res_avg = np.zeros((N,), dtype='f8')    # Outflow from reservoir
    Qin_Channel_avg = np.zeros((N,), dtype='f8') # Inflow to Channel
    Qout_channel_avg = np.zeros((N,), dtype='f8')# Outflow from Channel
    Sending= np.zeros((N,), dtype='f8')# Outflow from Channel

    # inverse of residence time
    tauinv = np.divide(ChV, L)   
    dtinv = 1.0 / dt
    # reshape the water management input data
    ppose = np.reshape(ppose,(N,))
    cpa   = np.reshape(cpa,(N,))
    unit_conversion_demand = area*1e3 / (nday*24*3600)
    monthly_demand   = np.reshape(np.multiply(wdirr,unit_conversion_demand),(N,)) #downstream  demand: m^3/s
    mean_demand  = np.reshape(np.multiply(irrmean,unit_conversion_demand),(N,))   #downstream mean demand:  m^3/s
    mtifl = np.reshape(mtifl,(N,))

    # classification based on purpose
    #condH = np.where(ppose ==1)[0] # Hydropower reservoir cells	
    #condI = np.where(ppose ==2)[0] # irrigation reservoir cells
    #condF = np.where(ppose ==3)[0] # flood reservoir cells
    usgrid_ppose = np.squeeze(np.array(pd.DataFrame(ppose[grdc_us_grids])))
    grdc_us_grids_ = np.squeeze(np.array(pd.DataFrame(grdc_us_grids)))
    condH= grdc_us_grids_[(usgrid_ppose==1)] # Hydropower reservoir cells	
    condI= grdc_us_grids_[(usgrid_ppose==2)] # irrigation reservoir cells
    condF= grdc_us_grids_[(usgrid_ppose==3)] # flood reservoir cells
    cond_all= grdc_us_grids_[(usgrid_ppose>0)]

	# grid water consumption
    if res_flag==1: 
        # water consumption
        qq = q - WConsumption
        # unmet consumption
        Wunmet = qq < 0
        # set excess runoff to zero for unmet cells
        qq[Wunmet] = 0 
    else:
        qq = q.copy()
    erlateral = (qq * area) * (1e3) / (nday * 24 * 3600)    # q -> erlateral: mm/month to m^3/s  

    # routing at 3hr time step 
    for t in range(nt):
        # Channel Storage Balance      
        # compute trial steps for F, S
        F = S * tauinv     # vector dot multiply
        dSdt= UM.dot(F) + erlateral  # vector
        Qin = UP.dot(F) + erlateral  # vector  

        Sin = S.copy()        
        # Have to check for any flows that will be greater than the actual amount of water available.
        Sx = ((dSdt * dt)+S < 0 )     # logic

        if Sx.any():
            # For cells with excess flow, let the flow be all the water available:  inbound + lateral + storage.
            # Since dSdt = inbound + lateral - F, we can get the new, adjusted value by adding to dSdt the F
            #  we calculated above and adding to that S / dt
            F[Sx] = Qin[Sx] + (S[Sx] * dtinv)

            # The new F is all the inflow plus all the storage, so the final value of S is zero.
            dSdt[Sx] = -S[Sx]* dtinv
            S[Sx]    = 0
            
            # For the rest of the cells, recalculate dSdt using the updated fluxes
            # (some of them will have changed due to the changes above.
            Sxn = np.logical_not(Sx)
            dSdt[Sxn] = (UM.dot(F))[Sxn] + erlateral[Sxn]
            S[Sxn] += dSdt[Sxn] * dt

            # NB: in theory we should iterate this procedure until there are no
            # further cells with excess flow, but there is no guarantee that
            # the iteration process will converge.
        else:
            # No excess flow, so use the forward-Euler formula for all cells
            S += (dSdt * dt)

        # results from channel 
        Sout = S.copy()
        Qout = F.copy()             
        Qout_channel_avg += Qout.copy()
        Qin_Channel_avg += Qin.copy()

        # start of water management subroutines
        if res_flag==1: 
		    ### Reservoir Release and Storage Balance 
            Qin_resv  = F.copy()
            Rres      = F.copy()			
			# Reservoirs Release Calculation
			## Flood Control Reservoirs
            Rres[condF]  = flood_control_reservoir(cpa, condF, Qin_resv, Sini_byr, mtifl, alpha, dt)
           

			## Irrigation Reservoirs
            Rres[condI]  = irrigation_reservoir(cpa, condI, Qin_resv, Sini_byr, mtifl, monthly_demand, mean_demand, alpha, dt)


			## Hydropower Reservoirs
            #Rres[condH]  = hydropower_reservoir(cpa[condH], Qin_resv[condH], Sini_resv[condH], Release_policy, maxTurbineFlow[condH], dt)
            #HPrelease(QQ, r_policy_all, q_max, cap, Sin)
            Rres[condH]  = HPrelease(Qin_resv[condH], Release_policy, maxTurbineFlow[condH], cpa[condH],  Sini_resv[condH], alpha)
            

            # Reservoir Water Balance         
            Qout_resv = Rres.copy()			
            Qout_resv[cond_all],  Sending[cond_all]  = reservoir_water_balance(Qin_resv, Rres, 
																			   Sini_resv, cpa, 
																			   mtifl, alpha, dt, cond_all)   

            #final
            F = Qout_resv.copy()			
            Favg += F
            Qin_res_avg +=  Qin_resv.copy()
            Qout_res_avg += Qout_resv.copy()
            Sini_resv = Sending.copy()
        else:
            Favg += F

    # monthly stream flow	
    Favg /= nt
    # monthly channel inflow and out flow
    Qin_Channel_avg /= nt    
    Qout_channel_avg /= nt    
    # monthly reservoir inflow and release   			
    Qin_res_avg /= nt
    Qout_res_avg /= nt                               

    return S , Favg, F ,Qin_Channel_avg, Qout_channel_avg,  Qin_res_avg, Qout_res_avg, Sending
        


def downstream(coord, flowdir, settings):
    """Generate downstream cell ID matrix"""

    gridmap = np.zeros((settings.ngridrow, settings.ngridcol), dtype=int, order='F')
    # Insert grid cell ID to 2D grid index position
    gridmap[coord[:, 4].astype(int) - 1, coord[:, 3].astype(int) - 1] = coord[:, 0]

    gridlen = coord.shape[0]
    # ilat and ilon are the row and column numbers for each working cell in the full grid
    ilat = coord[:, 4].astype(int) - 1
    ilon = coord[:, 3].astype(int) - 1

    fdlat, fdlon = make_flowdirgrid(ilat, ilon, flowdir, gridlen)

    # Fix cells that are pointing off the edge of the full grid, if any.
    # Wrap the longitude
    bad = (fdlon < 0) | (fdlon > (settings.ngridcol - 1))
    fdlon[bad] = np.mod(fdlon[bad] + 1, settings.ngridcol)
    # Set bad latitudes to point at self, which will be detected as an outlet below.
    bad = (fdlat < 0) | (fdlat > (settings.ngridrow - 1))
    fdlat[bad] = ilat[bad]
    fdlon[bad] = ilon[bad]

    # Get index of the downstream cell.
    tmp = np.ravel_multi_index((fdlat, fdlon), (settings.ngridrow, settings.ngridcol), order='F')
    tmpGM = np.ravel(gridmap, order='F')
    dsid = tmpGM[tmp]

    # Mark cells that are outlets.  These are cells that point to a cell
    # outside the working set (i.e., an ocean cell) and cells that point at
    # themselves (i.e., has no flow direction).
    ocoutlet = (dsid == 0)
    selfoutlet = (dsid == coord[:, 0])
    dsid[ocoutlet | selfoutlet] = -1

    return dsid


def upstream(coord, downstream, settings):
    """Return a matrix of ngrid x 9 values.
    For each cell, the first 8 values are the cellIDs neighbor cells.
    The 9th is the number of neighbor cells that actually flow into the center cell.
    The neighbor cells are ordered so that the cells that flow into the center cell come first.
    Thus, if these are the columns in a row:

    id1 id2 id3 id4 id5 id6 id7 id8 N

    if N==3, then id1, id2, ad id3 flow into the center cell; the others don't.
    Many cells will not have a full complement of neighbors. These missing neighbors are given the ID 0 """

    gridmap = np.zeros((settings.ngridrow, settings.ngridcol), dtype=int, order='F')
    # Insert grid cell ID to 2D grid index position
    gridmap[coord[:, 4].astype(int) - 1, coord[:, 3].astype(int) - 1] = coord[:, 0]  # 1-67420

    glnrow = coord.shape[0]
    upcells = np.zeros((glnrow, 8), dtype=int)
    isupstream = np.zeros((glnrow, 8), dtype=bool)

    # Row and column offsets for the 8 possible neighbors.
    rowoff = [-1, -1, -1, 0, 0, 1, 1, 1]
    coloff = [-1, 0, 1, -1, 1, -1, 0, 1]

    for nbr in range(8):
        r = coord[:, 4].astype(int) - 1 + rowoff[nbr]
        c = coord[:, 3].astype(int) - 1 + coloff[nbr]
        goodnbr = (r >= 0) & (c >= 0) & (r <= settings.ngridrow - 1) & (c <= settings.ngridcol - 1)

        if goodnbr.any():
            tmp = np.ravel_multi_index(
                (r[goodnbr], c[goodnbr]), (settings.ngridrow, settings.ngridcol), order='F')
            tmpGM = np.ravel(gridmap, order='F')
            upcells[goodnbr, nbr] = tmpGM[tmp]

        # Some cells have a zero in the grid map, indicating they are not being
        # tracked (they are ocean cells or some such.  Reset the mask to
        # reflect only the cells that are 'real' neighbors.
        goodnbr = np.logical_not(upcells[:, nbr] == 0)

        # Determine which cells flow into the center cell
        goodCoord = coord[goodnbr, 0].astype(int)
        goodDownstream = downstream[upcells[goodnbr, nbr] - 1]
        isupstream[goodnbr, nbr] = np.equal(goodCoord, goodDownstream)

    # Sort the neighbor cells so that the upstream ones come first.
    try:
        permvec = np.argsort(-isupstream)  # Sort so that True values are first
    except TypeError:
        # for newer versions of NumPy
        permvec = np.argsort(~isupstream)

    isupstream.sort()
    isupstream = isupstream[:, ::-1]

    ndgrid, _ = np.mgrid[0:glnrow, 0:8]  # Get necessary row adder
    permvec = permvec * glnrow + ndgrid

    tmpU = upcells.flatten('F')
    tmpP = permvec.flatten('F')
    tmpFinal = tmpU[tmpP]
    tmpFinal = tmpFinal.reshape((glnrow, 8), order='F')

    # Count the number of upstream cells.
    cellCount = np.zeros((glnrow, 1), dtype=int)
    cellCount[:, 0] = np.sum(isupstream, axis=1)
    upcells = np.concatenate((tmpFinal, cellCount), axis=1)

    return upcells

def upstream_genmatrix(upid):
    """Generate a sparse matrix representation of the upstream cells for each cell.
    The RHS of the ODE for channel storage S can be writen as
    dS/dt = UP * F + erlateral - S / T
    Since the instantaneous channel flow, F = S / T, this is the same as:
    dS/dt = [UP - I] S / T + erlateral
    This function returns UM = UP - I
    The second argument is the Jacobian matrix, J."""

    N = upid.shape[0]

    # Preallocate the sparse matrix.
    # Since we know that each cell flows into at most one other cell (some don't flow into any),
    # we can be sure we will need at most N nonzero slots.
    ivals = np.zeros((N,), dtype=int)
    jvals = np.zeros((N,), dtype=int)
    lb = 0  # Lower bound: the first index for each group of entries

    for i in range(N):
        numUp = upid[i, 8]  # Number of upstream cells for the current cell
        if numUp > 0:  # Skip if no upstream cells
            ub = lb + numUp
            jvals[lb:ub] = upid[i, 0:numUp]
            ivals[lb:ub] = i + 1
            lb = ub

    data = np.ones_like(ivals[0:ub])
    row = ivals[0:ub] - 1
    col = jvals[0:ub] - 1
    
    UP = sparse.coo_matrix((data, (row, col)), shape=(N, N)) 
    UM = sparse.coo_matrix((data, (row, col)), shape=(N, N)) - sparse.eye(N, dtype=int)

    return UM, UP



def make_flowdirgrid(ilat, ilon, flowdir, gridlen):
    # These are the bitwise and values of all the flow codes that lead in each
    # of the four directions.  'Up' and 'down' refer to directions in our grid
    # (i.e., they add to lat for 'up' and subtract for 'down')
    rt = 1 + 2 + 2 ** 7
    lt = 2 ** 3 + 2 ** 4 + 2 ** 5
    up = 2 ** 5 + 2 ** 6 + 2 ** 7
    dn = 2 + 2 ** 2 + 2 ** 3

    # Calculate the offset
    flwdr = np.copy(flowdir)
    flwdr[flowdir == -9999.] = 0
    flwdr = flwdr.astype(int)

    fdlat = np.zeros((gridlen,), dtype=int)
    fdlon = np.zeros((gridlen,), dtype=int)
    fdlat[(dn & flwdr) != 0] = -1
    fdlat[(up & flwdr) != 0] = 1
    fdlon[(rt & flwdr) != 0] = 1
    fdlon[(lt & flwdr) != 0] = -1

    # Apply the offset to latitude and longitude
    fdlat = fdlat + ilat
    fdlon = fdlon + ilon

    return fdlat, fdlon


######################################################################
#                       NEW SUBROUTINES                              #
######################################################################+
# channel water balance error
def channel_water_balance_error(Qin_, Qout_, Sin_, Sout_, dt, us_grids):
    """Computes channel water balance error, acceptable erro is <=np.finfo(np.float32).eps"""             
    
    Snorm = 0.5*(Sout_ + Sin_)
    DSdt_Q = (Qin_ - Qout_)*dt
    DSdt_S = (Sout_ - Sin_)
    DS_diff = np.abs(DSdt_Q - DSdt_S)
    Wbalance_relative_error = np.divide(DS_diff,  Snorm)    
    Wbalance_relative_error[(Snorm < 1)] = 0   
    if np.max(Wbalance_relative_error) > np.finfo(np.float32).eps:
        us_grids_pd = np.squeeze(np.array(pd.DataFrame(us_grids)))
        mm = (Wbalance_relative_error==np.max(Wbalance_relative_error))
        
        print("Error: Channel water balance violated at XanthosID: {}".format(us_grids_pd[mm])) 
        print("Qin: {}".format(Qin_[mm]))  
        print("Qout:{}".format(Qout_[mm]))  
        print("Sin: {}".format(Sin_[mm]))  
        print("Sout:{}".format(Sout_[mm])) 
        print("Error: {}".format(Wbalance_relative_error[mm]))  
         
        import sys
        sys.exit()     
    return np.max(Wbalance_relative_error)   


# reservoir water balance error
def reservoir_water_balance_error(Qin_, Qout_, Sin_, Sout_, dt, us_grids):
    """Computes reservoir water balance error, acceptable erro is <=np.finfo(np.float32).eps"""      
    Snorm = 0.5*(Sout_ + Sin_)
    DSdt_Q = (Qin_ - Qout_)*dt
    DSdt_S = (Sout_ - Sin_)
    DS_diff = np.abs(DSdt_Q - DSdt_S)
    Res_Wbalance_relative_error = np.divide(DS_diff,  Snorm)    
    Res_Wbalance_relative_error[(Snorm < 1)] = 0   
    if np.max(Res_Wbalance_relative_error) > np.finfo(np.float32).eps:
        us_grids_pd = np.squeeze(np.array(pd.DataFrame(us_grids)))
        mm = np.where(Res_Wbalance_relative_error==np.max(Res_Wbalance_relative_error))[0]
        
        print("Error: Reservoir water balance violated at XanthosID: {}".format(us_grids_pd[mm])) 
        print("Qin: {}".format(Qin_[mm]))  
        print("Qout:{}".format(Qout_[mm]))  
        print("Sin: {}".format(Sin_[mm]))  
        print("Sout:{}".format(Sout_[mm])) 
        print("Error: {}".format(np.max(Res_Wbalance_relative_error) ))  
         
        import sys
        sys.exit()     
    return np.max(Res_Wbalance_relative_error)   
     
   


      
######################################################################
# WATER MANAGEMENT 
# Irrigation Release
def irrigation_reservoir(cpa, cond_ppose, qin, Sini, mtifl,
                         wdirr, irrmean, alpha, dt):
    """Computes release from irrigation reservoirs"""  
    #irrigation Reservoirs
    #condI = np.where((cpa != 0) & (ppose ==2))[0] # irrigation reservoir cells

    # initialization
    Nx = len(cond_ppose)
    Rprovisional = np.zeros([Nx,]) #Provisional Release
    Rirrg_final = np.zeros([Nx,])       #Final Release

    # water management
    monthly_demand = wdirr[cond_ppose] #downstream  demand: m^3/s
    mean_demand  = irrmean[cond_ppose] #downstream mean demand:  m^3/s
    mtifl_irr = mtifl[cond_ppose]      #mean flow:  m^3/s
    cpa_irr = cpa[cond_ppose]          #capacity: 1e6 m^3
    qin_irr = qin[cond_ppose]
    Sbegnining_ofyear = Sini[cond_ppose]

    #****** Provisional Release *******
    # mean demand & annual mean inflow
    m = mean_demand - (0.5 * mtifl_irr)  
    cond1 = np.where(m >= 0 )[0] # m >=0 ==> dmean > = 0.5*annual mean inflow
    cond2 = np.where(m < 0)[0]   # m < 0 ==> dmean <  0.5*annual mean inflow    

    # Provisional Release 
    demand_ratio = np.divide(monthly_demand[cond1] , mean_demand[cond1])
    Rprovisional[cond1] = np.multiply(0.5*mtifl_irr[cond1],  (1 + demand_ratio)) # Irrigation dmean >= 0.5*imean        
    Rprovisional[cond2] = mtifl_irr[cond2] + monthly_demand[cond2] - mean_demand[cond2]  # Irrigation dmean < 0.5*imean   

    #******  Final Release ****** 
    # capacity & annual total infow
    c = np.divide(cpa_irr, ( mtifl_irr * 365 * 24 * 3600))
    cond3 = np.where(c >= 0.5)[0] #c = capacity/imean >= 0.5
    cond4 = np.where(c < 0.5)[0]  #c = capacity/imean < 0.5 

    # c = capacity/imean >= 0.5
    Krls = np.divide(Sbegnining_ofyear,(alpha * cpa_irr)) 
    Rirrg_final[cond3] = np.multiply(Krls[cond3], mtifl_irr[cond3])
    # c = capacity/imean < 0.5  
    temp1 = (c[cond4]/0.5)**2
    temp2 = np.multiply(temp1,Krls[cond4])
    temp3 = np.multiply(temp2,Rprovisional[cond4])      
    temp4 = np.multiply((1 - temp1), qin_irr[cond4])
    Rirrg_final[cond4] = temp3 + temp4

    return Rirrg_final 

# Flood Control Release
def flood_control_reservoir(cpa, cond_ppose, qin, Sini, mtifl, alpha, dt):
    """Computes release from flood control reservoirs"""  
    #flood Reservoirs
    # initialization
    Nx = len(cond_ppose)
    Rprovisional = np.zeros([Nx,]) #Provisional Release
    Rflood_final = np.zeros([Nx,])       #Final Release

    # water management
    mtifl_flood = mtifl[cond_ppose]      #mean flow:  m^3/s
    cpa_flood = cpa[cond_ppose]          #capacity:   m^3
    qin_flood = qin[cond_ppose]          #mean flow:  m^3/s
    Sbegnining_ofyear = Sini[cond_ppose] #capacity:   m^3

    # Provisional Release 
    Rprovisional = mtifl_flood.copy()

    # Final Release 
    # capacity & annual total infow
    c = np.divide(cpa_flood, ( mtifl_flood * 365 * 24 * 3600))
    cond1 = np.where(c >= 0.5)[0] #c = capacity/imean >= 0.5
    cond2 = np.where(c < 0.5)[0]  #c = capacity/imean < 0.5 

    # c = capacity/imean >= 0.5
    Krls = np.divide(Sbegnining_ofyear,(alpha * cpa_flood)) 
    Rflood_final[cond1] = np.multiply(Krls[cond1], mtifl_flood[cond1])
    # c = capacity/imean < 0.5  
    temp1 = (c[cond2]/0.5)**2
    temp2 = np.multiply(temp1,Krls[cond2])
    temp3 = np.multiply(temp2,Rprovisional[cond2])      
    temp4 = np.multiply((1 - temp1), qin_flood[cond2])
    Rflood_final[cond2] = temp3 + temp4


    return Rflood_final  
 		
         
def GetLevel(max_depth, head, surface_area, capac, cap_live, V):  
    """Computes storage level for hydropower reservoirs"""     

    if np.isnan(max_depth) == True:
        c = (np.sqrt(2) / 3 )* ((surface_area * 1e6) ** (3/2)) / (capac * 1e6)
        y = (6 * V / (c**2)) ** (1 / 3)
        yconst = head - y
        if (yconst < 0):
            cap_live = np.nanmin([cap_live, capac - (((-yconst) ** 3) * (c ** 2 / 6 / 1e6))])
    else:
        c = 2 * capac / (max_depth * surface_area)
        y = max_depth * (V / (capac* 1e6))**(c / 2)
        yconst = head - max_depth      
        if (yconst < 0):
            cap_live = np.nanmin([cap_live, capac- ((-yconst /max_depth) ** (2 /c) * capac)])

    return y, yconst, cap_live


# Reservoir water balance
def reservoir_water_balance(Qin, Qout, Sin, cpa, mtifl, alpha, dt ,  resrvoirs_indx):
    """Re-adjusts release for environmental flow (if necessary) and 
          Computes the storage level after release for all types of reservoirs"""     
	# inputs
    Qin_ = Qin[resrvoirs_indx]
    Qout_= Qout[resrvoirs_indx]
    Sin_ = Sin[resrvoirs_indx]
    cpa_ = cpa[resrvoirs_indx]
    mtifl_= mtifl[resrvoirs_indx]

    # final storage and release initialization	
    Nx = len(resrvoirs_indx)


    # environmental flow   
    diff_rt = Qout_ - (mtifl_*0.1)
    indx_rt = np.where(diff_rt < 0)[0]
    Qout_[indx_rt] = 0.1*mtifl_[indx_rt]

    # storage  
    dsdt_resv = (Qin_ - Qout_)*dt
    Stemp = Sin_ + dsdt_resv


    Rfinal = np.zeros([Nx,])    # final release
    Sfinal = np.zeros([Nx,])    # final storage
    # condition a : storage > capacity
    Sa = (Stemp > (alpha*cpa_))
    if Sa.any():
        Sfinal[Sa] = alpha*cpa_[Sa]
        Rspill     = (Stemp[Sa] - (alpha*cpa_[Sa]))/dt
        Rfinal[Sa] = Qout_[Sa] + Rspill

    # condition b : storage <= 0
    Sb = (Stemp < 0)
    if Sb.any():    
        Sfinal[Sb] = 0
        Rfinal[Sb] = (Sin_[Sb]/dt) + Qin_[Sb]

    # condition c : 25% capacity < S < capacity
    Sc = ((Stemp > 0) & (Stemp <= alpha*cpa_))
    if Sc.any():    
        Sfinal[Sc] = Stemp[Sc]
        Rfinal[Sc] = Qout_[Sc] 

    return Rfinal,  Sfinal 

def grdc_stations_upstreamgrids(upid, gridID):
    operate = [gridID]
    results = [gridID]
    while not (len(operate)==0) :
        grd_us = operate[0]
        immediate_us = upid[grd_us,0:upid[grd_us,8]]
        for ii in range(len(immediate_us)):
            results.append(immediate_us[ii]-1)
            operate.append(immediate_us[ii]-1)
        operate.remove(operate[0])
    
    contributing_grids = results

    return contributing_grids       



def release_functions(res, dataflow, res_data, alpha):
    # factors
    secs_in_month = 2629800  # number of seconds in an average month
    cumecs_to_Mm3permonth = 2.6298  # m3/s to Mm3/month
    sww = 9810  # specific weight of water (N/m^3)
    hours_in_year = 8766  # number of hours in a year
    mwh_to_exajoule = 3.6 * (10 ** -9)  # megawatts to exajoules
    mths = 12  # months in year
    # inflow
    QQ = dataflow[res,:]
    #reserv
    ########################
    cap = alpha*res_data["CAP"][res]#MCM
    cap_live = alpha*res_data["CAP"][res] #MCM  
    installed_cap = res_data["ECAP"][res]
    q_max = res_data["FLOW_M3S"][res] * cumecs_to_Mm3permonth
    efficiency = 0.9
    surface_area = res_data["AREA"][res]
    max_depth = res_data["DAM_HGT"][res]
    head = max_depth
    if np.isnan(head) == True:
        head = installed_cap / (efficiency * sww * (q_max / secs_in_month))
    if np.isnan(q_max)==True:
        qmax = (installed_cap / (efficiency * sww * head)) * secs_in_month
    ######################
    r_disc = 10
    s_disc = 1000
    s_states = np.linspace(0, cap, s_disc+1)  # storage discretized to 1000 segments for stoch. dyn. prog.
    r_disc_x = np.linspace(0, q_max, r_disc+1)
    m = mths
    q_Mm3 = QQ*cumecs_to_Mm3permonth
    inflow = pd.DataFrame(q_Mm3)
    # array setup
    start_date = date(1971, 1, 1)  # Get start date for simulation "M/YYYY"
    Q_month_mat = inflow.set_index(pd.period_range(start_date, periods = len(inflow), freq = "M"))
    Q_disc = np.array((0, 0.2375, 0.4750, 0.7125, 0.95, 1))
    Q_probs = np.diff(Q_disc)  # probabilities for each q class
    Q_class_med = Q_month_mat.groupby(Q_month_mat.index.month).quantile(Q_disc[1:6] - (Q_probs / 2))
    # set up empty arrays to be populated
    shell_array = np.zeros(shape=(len(Q_probs), len(s_states), len(r_disc_x)))
    rev_to_go = np.zeros(len(s_states))
    #Bellman = np.zeros([len(s_states), m])
    r_policy_test = np.zeros([len(s_states), m])
    # work backwards through months of year (12 -> 1) and repeat till policy converges

    while True:
        r_policy = np.zeros([len(s_states), m])
        for t in range(m, 0, -1):
            r_cstr = shell_array + np.array(Q_class_med.loc[t, slice(None)][0])[:, np.newaxis][:, np.newaxis] + \
                    shell_array + s_states[:, np.newaxis]  # constrained releases
            r_star = shell_array + r_disc_x   # desired releases       
            s_nxt_stage = r_cstr - r_star
            s_nxt_stage[s_nxt_stage < 0] = 0
            s_nxt_stage[s_nxt_stage > cap] = cap
            y, yconst, cap_live = GetLevel(max_depth, head, surface_area, cap, cap_live, (s_nxt_stage + s_states[:, np.newaxis])*1e6 / 2)
            h_arr = y + yconst
            #^^get head for all storage states for revenue calculation
            rev_arr = np.multiply(h_arr, r_star)  # revenue taken as head * release
            implied_s_state = np.around(1 + (s_nxt_stage / cap) * (len(s_states) - 1)).astype(np.int64)
            #^^implied storage is the storage implied by each release decision and inflow combination    
            rev_to_go_arr = rev_to_go[implied_s_state - 1]
            max_rev_arr = rev_arr + rev_to_go_arr
            max_rev_arr_weighted = max_rev_arr * np.array(Q_probs)[:, np.newaxis][:, np.newaxis]
            max_rev_arr_weighted[r_star > r_cstr] = float("-inf")  # negative rev to reject non-feasible release
            max_rev_expected = max_rev_arr_weighted.sum(axis=0)
            rev_to_go = max_rev_expected.max(1)
            r_policy[:, t - 1] = np.argmax(max_rev_expected, 1) 
        pol_test = float(sum(sum(r_policy == r_policy_test))) / (m *len(s_states))
        r_policy_test = r_policy  # re-assign policy test for next loop test
        #print(pol_test)
        if pol_test >= 0.99:
            break

    return r_policy


def HPrelease(QQ, r_policy_all, q_max, capp, Sinn, alpha):
    r_disc = 10
    s_disc = 1000
    cumecs_to_Mm3permonth = 2.6298  # m3/s to Mm3/month
    cap = alpha*capp*1e-6 #MM3
    Sin = Sinn*1e-6 #MM3    
    Nx  = len(cap)    
    QQcomputed = np.zeros([Nx,])
    Shp = np.zeros([Nx,])
    for ii in range(Nx):
        s_states = np.linspace(0, cap[ii], s_disc+1)  # storage discretized to 1000 segments for stoch. dyn. prog.
        r_disc_x = np.linspace(0, q_max[ii]* cumecs_to_Mm3permonth, r_disc+1) 
        Sct = Sin[ii]
        Qx = QQ[ii]*cumecs_to_Mm3permonth
        r_policy = r_policy_all[ii,:].astype(np.int64)   
        s_diff = np.abs(s_states - Sct)
        S_state = np.where(s_diff==np.min(s_diff)) #np.min(abs(s_states - S[ii]))
        indxt = r_policy[S_state][0]
        R = r_disc_x[indxt]
        R = min(R, Sct + Qx)# - (capacity - capacity_live))
        R_rec = R
        Spill = 0

        s_beggining = Sct 
        dsdt_resv = (Qx-R)
        Stemp = s_beggining + dsdt_resv
        if (Stemp > cap[ii]):
            Shp[ii] = cap[ii]
            Spill = Stemp - cap[ii] 
        else:
            if (Stemp < 0):
                Shp[ii] = 0
                R_rec = max(0, s_beggining + Qx)
            else:
                Shp[ii] = Stemp
        QQcomputed[ii] = (R_rec + Spill)/cumecs_to_Mm3permonth  # Mm3/month to m3/s 
 

    return QQcomputed