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
Lake and Water Management Components
@last update   07/27/2023
@authors: Guta Wakbulcho Abeshu : gwabeshu@uh.edu
          University of Houston
          HongYi Li : hli57@uh.edu
          University of Houston
"""

import pandas as pd
import numpy as np
import scipy.sparse as sparse
from datetime import date
from xanthos.reservoirs import WaterManagement


def streamrouting(L, S0, F0, ChV, q, area, nday, dt, UM, UP, Sini_byr, wdirr,
                  irrmean, mtifl, ppose, cpa, Release_policy, maxTurbineFlow,
                  WConsumption, alpha, Sini_resv, res_flag, grdc_us_grids):
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
    UM:   connection matrix (see notes in upstream_genmatrix)= (N x N, sparse)
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
    Qin_Channel_avg = np.zeros((N,), dtype='f8')  # Inflow to Channel
    Qout_channel_avg = np.zeros((N,), dtype='f8')  # Outflow from Channel
    Sending = np.zeros((N,), dtype='f8')  # Outflow from Channel

    # inverse of residence time
    tauinv = np.divide(ChV, L)
    dtinv = 1.0 / dt
    # reshape the water management input data
    ppose = np.reshape(ppose, (N,))
    cpa = np.reshape(cpa, (N,))
    unit_conversion_demand = area*1e3 / (nday*24*3600)
    # downstream  demand: m^3/s
    monthly_demand = np.reshape(np.multiply(
                     wdirr, unit_conversion_demand), (N, ))
    # downstream mean demand:  m^3/s
    mean_demand = np.reshape(np.multiply(
                     irrmean, unit_conversion_demand), (N, ))
    mtifl = np.reshape(mtifl, (N,))

    # grid water consumption
    if res_flag == 1:
        # water consumption
        qq = q - WConsumption
        # unmet consumption
        Wunmet = qq < 0
        # set excess runoff to zero for unmet cells
        qq[Wunmet] = 0
    else:
        qq = q.copy()
    # q -> erlateral: mm/month to m^3/s
    erlateral = (qq * area) * (1e3) / (nday * 24 * 3600)

    # classification based on purpose
    usgrid_ppose = np.squeeze(np.array(pd.DataFrame(ppose[grdc_us_grids])))
    grdc_us_grids_ = np.squeeze(np.array(pd.DataFrame(grdc_us_grids)))
    cond_hpr = grdc_us_grids_[(usgrid_ppose == 1)]  # HP reservoir cells
    cond_irr = grdc_us_grids_[(usgrid_ppose == 2)]  # irrig reservoir cells
    cond_fld = grdc_us_grids_[(usgrid_ppose == 3)]  # flood reservoir cells
    cond_all = grdc_us_grids_[(usgrid_ppose > 0)]

    # initiate lake module
    lwb = LakeWaterBalance.LakeModel()
    # initiate water management module
    wm_ = WaterManagement.Reservoir(alpha=alpha, us_grids=grdc_us_grids)

    # routing at 3hr time step
    for t in range(nt):
        # Channel Storage Balance
        # compute trial steps for F, S
        F = S * tauinv     # vector dot multiply
        dSdt = UM.dot(F) + erlateral  # vector
        Qin = UP.dot(F) + erlateral  # vector

        # Sin = S.copy()
        # Have to check for any flows that will be greater
        # than the actual amount of water available.
        Sx = ((dSdt * dt) + S < 0)     # logic

        if Sx.any():
            # For cells with excess flow, let the flow be all
            # the water available:  inbound + lateral + storage.
            # Since dSdt = inbound + lateral - F, we can get the new,
            # adjusted value by adding to dSdt the F
            #  we calculated above and adding to that S / dt
            F[Sx] = Qin[Sx] + (S[Sx] * dtinv)

            # The new F is all the inflow plus all the storage,
            # so the final value of S is zero.
            dSdt[Sx] = -S[Sx] * dtinv
            S[Sx] = 0

            # For the rest of the cells, recalculate dSdt using
            # the updated fluxes
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
        # Sout = S.copy()
        Qout = F.copy()
        Qout_channel_avg += Qout.copy()
        Qin_Channel_avg += Qin.copy()

        # start of water management subroutines
        if res_flag == 1:
            Qin_resv = F.copy()
            Rres = F.copy()
            # Flood Control Reservoirs
            Rres[cond_fld] = wm_.compute_flood_control_res_release(
                                                cpa,
                                                cond_fld,
                                                Qin_resv,
                                                Sini_byr,
                                                mtifl,
                                                )
            # Irrigation Reservoirs
            Rres[cond_irr] = wm_.compute_irrigation_res_release(
                                               cpa,
                                               cond_irr,
                                               Qin_resv,
                                               Sini_byr,
                                               mtifl,
                                               monthly_demand,
                                               mean_demand,
                                               )
            # Hydropower Reservoirs
            Rres[cond_hpr] = wm_.compute_hydropower_res_release(
                                    Qin_resv[cond_hpr],
                                    Release_policy,
                                    maxTurbineFlow[cond_hpr],
                                    cpa[cond_hpr],
                                    Sini_resv[cond_hpr])

            # Reservoir Water Balance
            Qout_resv = Rres.copy()
            (Qout_resv[cond_all],
             Sending[cond_all]) = wm_.compute_reservoir_water_balance(
                                                        Qin_resv, Rres,
                                                        Sini_resv, cpa,
                                                        mtifl, dt,
                                                        cond_all)

            # final
            F = Qout_resv.copy()
            Favg += F
            Qin_res_avg += Qin_resv.copy()
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

    return (S, Favg, F, Qin_Channel_avg, Qout_channel_avg,
            Qin_res_avg, Qout_res_avg, Sending)


def downstream(coord, flowdir, settings):
    """Generate downstream cell ID matrix"""

    gridmap = np.zeros((settings.ngridrow,
                        settings.ngridcol), dtype=int, order='F')
    # Insert grid cell ID to 2D grid index position
    gridmap[coord[:, 4].astype(int) - 1,
            coord[:, 3].astype(int) - 1] = coord[:, 0]

    gridlen = coord.shape[0]
    # ilat and ilon are the row and column numbers for each working
    # cell in the full grid
    ilat = coord[:, 4].astype(int) - 1
    ilon = coord[:, 3].astype(int) - 1

    fdlat, fdlon = make_flowdirgrid(ilat, ilon, flowdir, gridlen)

    # Fix cells that are pointing off the edge of the full grid, if any.
    # Wrap the longitude
    bad = (fdlon < 0) | (fdlon > (settings.ngridcol - 1))
    fdlon[bad] = np.mod(fdlon[bad] + 1, settings.ngridcol)
    # Set bad latitudes to point at self, which will be
    # detected as an outlet below.
    bad = (fdlat < 0) | (fdlat > (settings.ngridrow - 1))
    fdlat[bad] = ilat[bad]
    fdlon[bad] = ilon[bad]

    # Get index of the downstream cell.
    tmp = np.ravel_multi_index(
            (fdlat, fdlon),
            (settings.ngridrow, settings.ngridcol), order='F')
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
    The 9th is the number of neighbor cells that
    actually flow into the center cell.
    The neighbor cells are ordered so that the cells
    that flow into the center cell come first.
    Thus, if these are the columns in a row:

    id1 id2 id3 id4 id5 id6 id7 id8 N

    if N==3, then id1, id2, ad id3 flow into the center cell; the others don't.
    Many cells will not have a full complement of neighbors.
    These missing neighbors are given the ID 0 """

    gridmap = np.zeros((settings.ngridrow,
                        settings.ngridcol), dtype=int, order='F')
    # Insert grid cell ID to 2D grid index position
    gridmap[coord[:, 4].astype(int) - 1,
            coord[:, 3].astype(int) - 1] = coord[:, 0]  # 1-67420

    glnrow = coord.shape[0]
    upcells = np.zeros((glnrow, 8), dtype=int)
    isupstream = np.zeros((glnrow, 8), dtype=bool)

    # Row and column offsets for the 8 possible neighbors.
    rowoff = [-1, -1, -1, 0, 0, 1, 1, 1]
    coloff = [-1, 0, 1, -1, 1, -1, 0, 1]

    for nbr in range(8):
        r = coord[:, 4].astype(int) - 1 + rowoff[nbr]
        c = coord[:, 3].astype(int) - 1 + coloff[nbr]
        goodnbr = (r >= 0) & (c >= 0) & (
                                        r <= settings.ngridrow - 1) & (
                                        c <= settings.ngridcol - 1)

        if goodnbr.any():
            tmp = np.ravel_multi_index(
                (r[goodnbr], c[goodnbr]), (settings.ngridrow,
                                           settings.ngridcol), order='F')
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
    """Generate a sparse matrix representation of the upstream cells for each
    cell. The RHS of the ODE for channel storage S can be writen as
    dS/dt = UP * F + erlateral - S / T
    Since the instantaneous channel flow, F = S / T, this is the same as:
    dS/dt = [UP - I] S / T + erlateral
    This function returns UM = UP - I
    The second argument is the Jacobian matrix, J."""

    N = upid.shape[0]

    # Preallocate the sparse matrix.
    # Since we know that each cell flows into at most
    # one other cell (some don't flow into any),
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
    UM = sparse.coo_matrix((
            data, (row, col)), shape=(N, N)) - sparse.eye(N, dtype=int)

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


def grdc_stations_upstreamgrids(upid, gridID):
    operate = [gridID]
    results = [gridID]
    while not (len(operate) == 0):
        grd_us = operate[0]
        immediate_us = upid[grd_us, 0:upid[grd_us, 8]]
        for ii in range(len(immediate_us)):
            results.append(immediate_us[ii]-1)
            operate.append(immediate_us[ii]-1)
        operate.remove(operate[0])

    contributing_grids = results

    return contributing_grids
