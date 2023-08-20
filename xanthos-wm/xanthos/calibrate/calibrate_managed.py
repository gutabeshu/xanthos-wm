"""
Calibrate the ABCD model with and without water management.
@authors:Guta Abeshu : gwabeshu@uh.edu
         University of Houston
         HongYi Li : hli57@uh.edu,
         University of Houston
License:  BSD 2-Clause, see LICENSE and DISCLAIMER files
Copyright (c) 2018, Battelle Memorial Institute
"""

import os
import sys
import warnings
import logging
from datetime import date
from itertools import product

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import spotpy
import xanthos.calibrate.calibrate_abcdm_firststage as calibrate_runoff
#import xanthos.routing.mrtm_managed as routing_mod
import xanthos.routing.mrtm_managed_lake as routing_mod
import xanthos.utils.general as helper
import xanthos.utils.math as umth
from scipy.stats import qmc
from xanthos.data_reader.data_abcd import DataAbcd
from xanthos.data_reader.data_calibration_managed import DataCalibrationManaged
from xanthos.data_reader.data_mrtm_managed import DataMrtmManaged
from xanthos.data_reader.data_reference import DataReference
from xanthos.runoff.abcd_managed import AbcdManaged
from xanthos.reservoirs import WaterManagement

warnings.filterwarnings('ignore')
logging.basicConfig(filename="Calibration.log", level=logging.INFO)


class CalibrateManaged:
    """Calibrate the ABCD model with and without water management.
    """

    ncell = 67420
    ngridrow = 360
    ngridcol = 720
    # Runoff parameters bound
    LB = 1e-2
    UB = 1 - LB
    # WM parameters bound
    LBwm1 = 0.1
    UBwm1 = 10.0
    PARAMETER_NAMES = ['a', 'b', 'd', 'm', 'beta', 'alpha']

    def __init__(self,
                 basin_num,
                 basin_ids,
                 basin_areas,
                 precip,
                 pet,
                 obs,
                 tmin,
                 runoff_spinup,
                 set_calibrate,
                 obs_unit,
                 out_dir,
                 router_func=None,
                 config_obj=None,
                 cal_observed=None,
                 ):
        """Initialize calibration data and parameters.
        :param basin_num:      basin number as an integer
        :param basin_ids:      an array of basin ids per grid cell that are
                               associated with the basin
        :param basin_areas:    an array of basin areas per grid cell that are
                               associated with the basin
        :param precip:         precipitation in mm/month
        :param pet:            PET in mm/month
        :param obs:            Observed runoff in mm/month
        :param tmin:           minimum temperature in degrees C
        :param runoff_spinup:  the number of months from time 0 in the input
                               data to use as spin up
        :param set_calibrate:  0 to calibrate to observed runoff, 1 to
                               calibrate to observed streamflow
        :param obs_unit:       the unit of the input data
        :param out_dir:        calibrated parameters output directory
        :param router_func:    objective function for calibrating routing
        """

        # get data from config
        self.start_year = config_obj.StartYear
        self.end_year = config_obj.EndYear
        self.nmonths = config_obj.nmonths
        self.reference_data = DataReference(config=config_obj)
        self.routing_spinup = config_obj.routing_spinup
        self.repetitions = config_obj.repetitions
        self.calib_algorithm_runoff = config_obj.cal_algrthm_runoff
        self.calib_algorithm_streamflow = config_obj.cal_algrthm_flow
        self.flow_distance_file = config_obj.flow_distance
        self.flow_direction_file = config_obj.flow_direction
        self.stream_velocity_file = config_obj.strm_veloc
        self.historical_mode = config_obj.HistFlag
        self.hist_channel_storage_file = config_obj.ChStorageFile
        self.hist_channel_storage_varname = config_obj.ChStorageVarName

        # grid data
        self.basin_num = basin_num
        self.basin_ids = basin_ids
        self.basin_areas = basin_areas
        self.precip = precip
        self.pet = pet
        self.obs = obs
        self.tmin = tmin
        self.runoff_spinup = runoff_spinup
        self.router_func = router_func
        self.set_calibrate = set_calibrate
        self.obs_unit = obs_unit
        self.out_dir = out_dir

        self.parameternames = CalibrateManaged.PARAMETER_NAMES
        # load calibration data
        self.calib_data = DataCalibrationManaged(
                            config_obj=config_obj,
                            cal_observed=cal_observed,
                            start_year=self.start_year,
                            end_year=self.end_year)

        # index for gauge station locations
        self.grdcData_info = self.calib_data.grdc_coord_index_file
        bgrdc_indx = np.where(
                     self.grdcData_info['basins'].values == self.basin_num)[0]
        if (len(bgrdc_indx) == 0) & (self.set_calibrate == 1):
            print("Error: GRDC data is not available for Basin: {}"
                  .format(self.basin_num))
            sys.exit()
        elif (len(bgrdc_indx) == 1) & ~(self.set_calibrate == 0):
            self.basin_grdc_indx = bgrdc_indx[0]
            self.grdc_xanthosID = self.grdcData_info.loc[
                                self.basin_grdc_indx]['xanthosID_new_adj'] - 1
            self.grdc_drainage_area = self.grdcData_info.loc[
                                    self.basin_grdc_indx]['area'] * 1e6  # m2

        # routing inputs: wdirr, irrmean, tifl, ppose, cpa,dscells
        self.wdirr = np.copy(self.calib_data.total_demand_mmpermonth)
        self.irrmean = np.mean(self.wdirr, axis=1)  # mean demand
        self.ppose = np.array(self.calib_data.purpose)
        self.cpa = np.array(self.calib_data.capacity) * 1e6  # m3
        self.installed_cap = np.array(self.calib_data.installed_cap)  # MW
        self.q_max = np.array(self.calib_data.maxtif_natural)  # m3/s
        self.surface_area = np.array(self.calib_data.surface_area)  # square km
        self.max_depth = np.array(self.calib_data.max_depth)  # square km

        self.WConsumption = np.array(self.calib_data.water_consumption)
        self.chs_ini = np.array(self.calib_data.ini_channel_storage)
        self.Initial_instream_flow = np.array(
                               self.calib_data.instream_flow_natural)
        self.SM = np.squeeze(np.array(self.calib_data.sm))
        self.res_data = pd.DataFrame([self.cpa * 1e-6,
                                      self.installed_cap,
                                      self.q_max,
                                      self.surface_area,
                                      self.max_depth]).transpose()
        self.res_data.columns = ["CAP",	"ECAP", "FLOW_M3S", "AREA", "DAM_HGT"]
        # routing data
        self.yr_imth_dys = helper.set_month_arrays(self.nmonths,
                                                   self.start_year,
                                                   self.end_year)
        self.map_index = umth.sub2ind(
                        [CalibrateManaged.ngridrow,
                            CalibrateManaged.ngridcol],
                        self.reference_data.coords[:, 4].astype(np.int64) - 1,
                        self.reference_data.coords[:, 3].astype(np.int64) - 1)
        self.routing_data = DataMrtmManaged(
                start_year=self.start_year,
                end_year=self.end_year,
                flow_distance_file=self.flow_distance_file,
                flow_direction_file=self.flow_direction_file,
                stream_velocity_file=self.stream_velocity_file,
                historical_mode=self.historical_mode,
                hist_channel_storage_file=self.hist_channel_storage_file,
                hist_channel_storage_varname=self.hist_channel_storage_varname)
        # contributing grids for grdc  station
        self.dsid = routing_mod.downstream(self.reference_data.coords,
                                           self.routing_data.flow_dir,
                                           CalibrateManaged)
        self.upid = routing_mod.upstream(self.reference_data.coords, self.dsid,
                                         CalibrateManaged)
        # upstream genmatrix matrix
        self.um, self.up = routing_mod.upstream_genmatrix(self.upid)

        # transpose data for use in the ABCD model
        if self.set_calibrate == 1:
            # grids upstream of grdc
            self.basin_idx = routing_mod.grdc_stations_upstreamgrids(
                             self.upid, self.grdc_xanthosID)
        else:
            self.basin_idx = np.where(self.basin_ids == self.basin_num)[0]

        self.bsn_areas = self.basin_areas[self.basin_idx]
        self.bsn_PET = self.pet[self.basin_idx]
        self.bsn_P = self.precip[self.basin_idx]
        # initial soil moisture
        self.bsn_SM = self.SM[self.basin_idx]

        # Minimum temperature is optional;
        # if not provided, the snow components
        # of the model is effectively removed, so remove the model parameter
        # if no tmin provided, just ensure it is larger than the rain threshold
        self.nosnow = self.tmin is None
        if self.nosnow:
            self.bsn_TMIN = None
            self.tminx = None
        else:
            self.bsn_TMIN = self.tmin[self.basin_idx]
            self.tminx = self.tmin

        # Unit conversion for runoff case
        # if self.obs_unit == "km3_per_mth":
        self.conversion = self.bsn_areas * 1e-6
        # Observation data for calibration
        if self.set_calibrate == 0:
            self.bsn_obs_runoff = np.squeeze(
                                  self.obs[np.where(
                                    self.obs[:, 0] == self.basin_num)[0], 3])
            self.bsn_obs = np.divide(np.squeeze(
                           self.obs[
                            np.where(self.obs[:, 0] == self.basin_num)[0],
                            3]) * 1e9, (self.yr_imth_dys[:, 2] * 24 * 3600))
            self.calib_length = int(0.7*len(self.bsn_obs_runoff))
            self.data_length = int(len(self.bsn_obs_runoff))
            self.bsn_obs_flow_calib = self.bsn_obs[0:self.calib_length]
            self.bsn_obs_flow_valid = self.bsn_obs[
                                          self.calib_length+1:self.data_length]
        else:
            self.bsn_obs = self.obs[np.where(
                        self.obs[:, 0] == self.basin_num)][: self.nmonths, 1]
            # calibration and validation data
            self.calib_length = int(0.5*len(self.bsn_obs))
            self.data_length = int(len(self.bsn_obs))
            self.bsn_obs_flow_calib = self.bsn_obs[0:self.calib_length]
            self.bsn_obs_flow_valid = self.bsn_obs[
                                       self.calib_length + 1:self.data_length]
            # basin observed runoff# mm/month
            self.bsn_obs_runoff = ((np.multiply(
                                  self.bsn_obs,
                                  self.yr_imth_dys[0:self.data_length, 2]) *
                                   (1e3 * 24 * 3600) / self.grdc_drainage_area)
                                   - np.mean(self.WConsumption[
                                    self.basin_idx, 0:self.data_length], 0))
        # residence time in hr
        Lst = self.routing_data.flow_dist
        self.Vst = self.routing_data.str_velocity
        self.Vst[self.Vst < 0.01] = 0.01
        grid_size = np.sqrt(self.reference_data.area) * 1000
        nn = np.where(Lst < grid_size)[0]
        Lst[nn] = grid_size[nn]
        self.Tr = np.divide(Lst, self.Vst) / 3600
        self.beta_local = 0.95*self.Tr/3

        self.flow_dist = self.routing_data.flow_dist
        grid_size = np.sqrt(self.reference_data.area) * 1e3  # grid size in m
        nn_grids = np.where(self.flow_dist < grid_size)[0]
        self.flow_dist[nn_grids] = grid_size[nn_grids]  # update flow distance
        self.flow_dir = self.routing_data.flow_dir

        # routing initializations
        # initial channel storage in m3
        self.chs_prev = np.squeeze(self.chs_ini)
        # initial channel flow in m3 /s
        self.instream_flow = np.squeeze(self.Initial_instream_flow)
        self.mtifl_natural = np.squeeze(self.Initial_instream_flow)
        # reservoir params
        usgrid_ppose = np.squeeze(np.array(pd.DataFrame(
                             self.ppose[self.basin_idx])))
        grdc_us_grids_ = np.squeeze(np.array(pd.DataFrame(self.basin_idx)))
        # Hydropower reservoir cells
        self.HPindex = grdc_us_grids_[(usgrid_ppose == 1)]
        self.us_resrv_idx = grdc_us_grids_[(usgrid_ppose > 0)]

        self.Main_UseHP = (self.ppose == 1)
        self.halfDegree_global_dfG = self.res_data[
                                     self.Main_UseHP].reset_index(drop=True)
        self.Main_UseHPXP = np.where(self.ppose == 1)[0]
        # best parameter from preceeding simulations
        self.best_params = None
        self.initial_cond = 0  # to run reservoir initialization
        self.dir_storage = self.out_dir + '/Sini_'

        # calibration result output path
        self.ModelPerformance = os.path.join(
                                  self.out_dir,
                                  f"basin_calibration_{self.basin_num}")

        # set up parameters  for first stage or runoff
        # set number of parameter combinations with latin hyper cube sampling
        self.l_bounds = [CalibrateManaged.LB, CalibrateManaged.LB,
                         CalibrateManaged.LB, CalibrateManaged.LB,
                         CalibrateManaged.LB]
        self.u_bounds = [CalibrateManaged.UB, 8 - CalibrateManaged.LB,
                         CalibrateManaged.UB, CalibrateManaged.UB,
                         CalibrateManaged.UB]
        # nsample = 10000 #number of sample per parameter
        # self.sample_params_set = np.zeros([nsample, len(self.l_bounds)])
        # for ii in range(len(self.l_bounds)):
        sampler_lhc = qmc.LatinHypercube(d=5, seed=42)
        sample_params = sampler_lhc.random(n=1000000)
        self.sample_params_set = np.squeeze(qmc.scale(
                                 sample_params,
                                 self.l_bounds, self.u_bounds))

        self.params_ro_lhc = [spotpy.parameter.List('a',
                              list(self.sample_params_set[:, 0])),
                              spotpy.parameter.List('b',
                              list(self.sample_params_set[:, 1])),
                              spotpy.parameter.List('c',
                              list(self.sample_params_set[:, 2])),
                              spotpy.parameter.List('d',
                              list(self.sample_params_set[:, 3])),
                              spotpy.parameter.List('m',
                              list(self.sample_params_set[:, 4])),
                              ]
        # two stage calibration case
        if (self.set_calibrate == 1):
            print("\tStarting The First Stage Parameter Selection :" +
                  "Runoff Parameters Selection")
            self.ro_params_selected = calibrate_runoff.calibrate_basin(
                                                self.start_year,
                                                self.end_year,
                                                self.pet,
                                                self.precip,
                                                self.tmin,
                                                self.SM,
                                                self.bsn_obs_runoff,
                                                self.basin_ids,
                                                self.basin_idx,
                                                self.nmonths,
                                                self.runoff_spinup,
                                                self.repetitions,
                                                self.calib_algorithm_runoff,
                                                self.ModelPerformance,
                                                self.params_ro_lhc,
                                                self.set_calibrate,
                                                self.conversion)

            # list of parameters values for second stage calibration
            print("Starting Second Stage Parameter Selection: Runoff"
                  "Routing Parameter Selection")
            # parameter setup for second stage calibration
            # Give possible beta values as a List
            self.wmp_beta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # Give possible alpha values as a List
            wmp_alpha = [0.85]
            wm_params = pd.DataFrame(product(self.wmp_beta, wmp_alpha))
            for ii in range(self.ro_params_selected.shape[0]):
                abcdm_parameters = pd.concat([pd.DataFrame(
                                    np.tile(
                                        self.ro_params_selected[ii, :],
                                        (len(self.wmp_beta), 1))),
                                        wm_params], 1)
                if ii == 0:
                    self.wm_abcdm_parameters = abcdm_parameters.copy()
                else:
                    self.wm_abcdm_parameters = pd.concat(
                                        [self.wm_abcdm_parameters,
                                            abcdm_parameters], 0).reset_index(
                                                drop=True)

            self.params_all = np.array(self.wm_abcdm_parameters)
            self.params_wm = [spotpy.parameter.List('a',
                              list(self.params_all[:, 0])),
                              spotpy.parameter.List('b',
                              list(self.params_all[:, 1])),
                              spotpy.parameter.List('c',
                              list(self.params_all[:, 2])),
                              spotpy.parameter.List('d',
                              list(self.params_all[:, 3])),
                              spotpy.parameter.List('m',
                              list(self.params_all[:, 4])),
                              spotpy.parameter.List('wmbeta',
                              list(self.params_all[:, 5])),
                              spotpy.parameter.List('wmalpha',
                              list(self.params_all[:, 6]))]

        elif (self.set_calibrate == -1):
            self.parametrs_abcdmba = self.calib_data.optimal_parameters
            self.params_wm = np.squeeze(self.parametrs_abcdmba[np.where(
                             self.parametrs_abcdmba[:, 7] == self.basin_num
                               )[0], 0:7])
    # parameter set up
    # def parameters(self):
        # '''Returns ABCD Params'''
        # if self.set_calibrate <= 0:
            # # parmateres from list created with lhc
            # params = self.params_ro_lhc
        # elif self.set_calibrate == 1:
            # # parmateres for wm
            # params = self.params_wm
        # return spotpy.parameter.generate(params)

    def parameters(self):
        '''Returns ABCD Params'''
        self.best_params = None
        if self.set_calibrate == 0:
            params_order = [0, 1, 2, 3, 4]
            params = [spotpy.parameter.Uniform(self.l_bounds[p],
                                               self.u_bounds[p])
                      for p in params_order]
        else:
            params = self.params_wm

        return spotpy.parameter.generate(params)

    # model simulation set up
    def simulation(self, pars):
        """ABCD model and mrtm routing model : this function provides
           simulated streamflow"""
        if self.set_calibrate == 0:
            he = AbcdManaged(pars=pars,
                             soil_water_initial=self.SM,  # [self.basin_idx],
                             pet=self.pet,  # [self.basin_idx, :],
                             precip=self.precip,  # [self.basin_idx, :],
                             tmin=self.tmin,  # [self.basin_idx, :],
                             basin_ids=self.basin_ids,  # [self.basin_idx],
                             process_steps=self.nmonths,
                             spinup_steps=self.runoff_spinup,
                             method="dist")
            he.emulate()
            rsim = he.rsim
            return np.nansum(
                            rsim[:, self.basin_idx] * self.conversion,
                            1)[0:self.calib_length]

        else:
            # runoff model
            he = AbcdManaged(pars=pars,
                             soil_water_initial=self.SM,
                             pet=self.pet,
                             precip=self.precip,
                             tmin=self.tmin,
                             basin_ids=self.basin_ids,
                             process_steps=self.nmonths,
                             spinup_steps=self.runoff_spinup,
                             method="dist")
            he.emulate()
            # runoff
            srunoff = he.rsim.T
            # self.runoff = np.load(
            # 'Basin_runoff_mmpermonth_pm_abcdm_gswp3_w5e5_1971_2010.npy')
            srunoff[srunoff < 0] = 0
            self.runoff = srunoff[:, 2:]
            # load routing data
            beta_basin = np.copy(self.beta_local)
            beta_basin[(self.Tr/pars[5]) > 3] = 1  # pars[5]
            # adjusted velocity
            self.str_velocity = np.multiply(np.copy(self.Vst), beta_basin)
            # Preallocation for variables to be returned from routing module
            # average channel flow m3/s
            self.Avg_ChFlow = np.zeros_like(self.precip)
            # channel storage   m3
            ChStorage = np.zeros_like(self.precip)
            # reservoir storage m3
            ResStorage = np.zeros_like(self.precip)
            # reservoir inflow outflow m3/s
            Qin_res_avg = np.zeros_like(self.precip)
            Qout_res_avg = np.zeros_like(self.precip)
            # 1001 = S discretization & 12 = nmonths/year
            basin_idx = np.arange(0, len(self.basin_ids), 1)
            HP_Release = np.zeros([len(self.basin_ids), 1001, 12])
            # routing time step
            self.routing_timestep = 3*3600  # seconds

            # initialize reservoir
            wm_ = WaterManagement.Reservoir(alpha=0.85, us_grids=basin_idx)
            simulation_length = self.nmonths

            for i in range(3):
                # print(ii)
                if i <= 1:
                    # ##################### WITHOUT RESERVOIR ################
                    # Reservoir flag
                    # 1 if with reservoir, 0 without reservoir
                    res_flag = 0
                    # routing time step
                    simulation_length = self.nmonths
                    # place holders
                    # initial reservoir storage at time t in m3
                    Sini = np.zeros_like(self.mtifl_natural)
                    # reservoir storage at beggining of the year in m3
                    res_prev = np.zeros_like(self.mtifl_natural)
                else:
                    # ##################### WITH RESERVOIR ####################
                    # Reservoir flag
                    res_flag = 1
                    # initial values

                    self.mtifl_natural = np.mean(self.Avg_ChFlow, 1)
                    if i == 1:
                        # routing time step
                        simulation_length = self.nmonths  # self.routing_spinup
                        # initial reservoir storage at time t in m3
                        Sini = 0.5 * np.squeeze(self.cpa)
                        # reservoir storage at beggining of the year in m3
                        res_prev = 0.5 * np.squeeze(self.cpa)
                    else:
                        # routing time step
                        simulation_length = self.nmonths
                        Sini_read = np.load(
                                     self.dir_storage +
                                     str(self.basin_num) + '.npy')
                        # initial reservoir storage at time t in m3
                        Sini = np.squeeze(Sini_read)
                        # reservoir storage at beggining of the year in m3
                        res_prev = np.squeeze(Sini_read)

                    # Hydropower reservoirs operation policy
                    if len(self.HPindex) > 0:
                        dataflowD = self.Avg_ChFlow[self.Main_UseHPXP, :]
                        # 1001 = S discretization & 12 = nmonths/year
                        HP_Release = np.zeros([
                            dataflowD.shape[0], 1001, 12])
                        for res in range(dataflowD.shape[0]):
                            HP_Release[
                                res, :, :] = wm_.compute_release_policies(
                                                res,
                                                dataflowD,
                                                self.halfDegree_global_dfG
                                                )

                # ##################### ROUTING ###########################
                for nm in range(simulation_length):
                    # lake forcing data
                    sl_forcing_data = forcing_data_lakes(
                                            self.sl_forcing_dict,
                                            self.yr_imth_dys[nm, 1] + 1,
                                            self.yr_imth_dys[nm, 2], nm)
                    ml_forcing_data = forcing_data_lakes(
                                            self.ml_forcing_dict,
                                            self.yr_imth_dys[nm, 1] + 1,
                                            self.yr_imth_dys[nm, 2], nm)
                    ll_forcing_data = forcing_data_lakes(
                                            self.ll_forcing_dict,
                                            self.yr_imth_dys[nm, 1] + 1,
                                            self.yr_imth_dys[nm, 2], nm)

                    sr = routing_mod.streamrouting(self.flow_dist,  # L
                                                   self.chs_prev,  # S0
                                                   self.instream_flow,  # F0
                                                   self.str_velocity,  # ChV
                                                   self.runoff[:, nm],  # q
                                                   self.basin_areas,  # area
                                                   self.yr_imth_dys[nm, 2],
                                                   self.routing_timestep,  # dt
                                                   self.um,  # um
                                                   self.up,  # up
                                                   Sini,  # Sini
                                                   self.wdirr[:, nm],  # wdirr
                                                   self.irrmean,  # irrmean
                                                   self.mtifl_natural,  # mtifl
                                                   self.ppose,  # ppose
                                                   self.cpa,  # cpa
                                                   HP_Release[:, :,
                                                              np.mod(0, 12)],
                                                   self.q_max,  # TurbineFlow
                                                   self.WConsumption[:, nm],
                                                   pars[6],  # alpha
                                                   res_prev,  # Sini_resv
                                                   res_flag,  # res_flag
                                                   basin_idx,
                                                   )

                    (
                        ChStorage[:, nm],
                        self.Avg_ChFlow[:, nm],
                        self.instream_flow,
                        self.Qin_Channel_avg,
                        self.Qout_channel_avg,
                        Qin_res_avg,
                        Qout_res_avg,
                        ResStorage[:, nm],
                        ) = sr

                    # update data
                    res_prev = ResStorage[:, nm]
                    # update the reservoir storage at beginning of year
                    if np.mod(nm, 12) == 11:
                        self.Sini = ResStorage[:, nm]

                    # update data channels storage with reservoir effect
                    DsDt_channel = ((Qin_res_avg - Qout_res_avg)
                                    * self.yr_imth_dys[nm, 2] * 24 * 3600)
                    self.chs_prev = ChStorage[:, nm] + DsDt_channel
                    self.chs_prev[self.chs_prev < 0] = 0

                # storage out if first run
                if i == 2:
                    np.save(
                        self.dir_storage +
                        str(self.basin_num) + '.npy',
                        res_prev)
                    self.initial_cond += 1

            if (self.set_calibrate == -1) | (self.set_calibrate == 1):
                simulated_flow = self.Avg_ChFlow[
                                  self.grdc_xanthosID,
                                  0: self.calib_length]
            else:
                simulated_flow = np.nanmean(
                                 self.Avg_ChFlow[
                                    self.basin_idx,
                                    0: self.calib_length], 0)

        return simulated_flow

    # @staticmethod
    def objectivefunction(self, simulation, evaluation):
        """Calculates Model Performance.
        Objective function to be minimized (if sceua /NSGAII is used) and
        maximized (all others)
        """
        # sceua requires minimization which will result in a negative KGE
        # method = self.calib_algorithm_streamflow
        # if (method == 'sceua') | (method == 'NSGAII'):
        #    multiplier = -1
        # else:
        multiplier = 1
        obj1 = -spotpy.objectivefunctions.kge(
                evaluation, simulation) * multiplier
        obj2 = spotpy.objectivefunctions.rrmse(
                evaluation, simulation)
        obj3 = spotpy.objectivefunctions.lognashsutcliffe(
                evaluation, simulation)

        return [obj1, obj2, obj3]

    def evaluation(self):
        """observed streamflow data"""
        if (self.set_calibrate == -1) | (self.set_calibrate == 1):
            self.obs_eval = self.bsn_obs_flow_calib
        else:
            self.obs_eval = self.bsn_obs_runoff[0:self.calib_length]

        return self.obs_eval

    def save(self, objectivefunctions, parameter, simulations):
        line = str(
            objectivefunctions) + ',' + str(
                parameter).strip('[]') + ',' + str(
                    simulations).strip('[]') + '\n'
        self.database.write(line)

    # calibration set up
    def calibrate_basin(self):
        """This function is to calibrate the distributed ABCD +
            water management model against the GRDC to
            obtain optimized parameters of ABCD(a, b, c, d, m) and
            Water management (beta and c)
        """
        # parallel ='seq' # Runs everthing in sequential mode
        np.random.seed(42)  # Makes the results reproduceable

        if (self.set_calibrate == -1) | (self.set_calibrate == 1):
            name_ext_flow = '_Flow_ObjF_monthlyKGE'
            dbname_dir = (
                         self.ModelPerformance +
                         self.calib_algorithm_streamflow +
                         name_ext_flow)
            # number of iterations
            if self.set_calibrate == 1:
                self.repetitions = 5  # self.params_all.shape[0]
            elif self.set_calibrate == -1:
                # 19 is combination of routing coeffitient and
                # the optimal runoff parameter set
                self.repetitions = 5  # len(self.wmp_beta)
            # algorithm
            if self.calib_algorithm_streamflow == 'sceua':
                sampler = spotpy.algorithms.sceua(self,
                                                  dbname=dbname_dir,
                                                  dbformat="csv",
                                                  dbappend=False,
                                                  save_sim=False,
                                                  parallel='mpi'
                                                  )
                sampler.sample(self.repetitions,
                               ngs=50, kstop=50,
                               peps=1e-2, pcento=1e-2)

            elif self.calib_algorithm_streamflow == 'NSGAII':
                n_pop = 10
                repetitions_nsgaii = int(self.repetitions / n_pop)
                sampler = spotpy.algorithms.NSGAII(self,
                                                   dbname=dbname_dir,
                                                   dbformat="csv",
                                                   dbappend=False,
                                                   save_sim=False,
                                                   parallel='mpi')
                sampler.sample(repetitions_nsgaii, n_obj=1, n_pop=n_pop)

            elif self.calib_algorithm_streamflow == 'mcmc':
                sampler = spotpy.algorithms.mcmc(self,
                                                 dbname=dbname_dir,
                                                 dbformat="csv",
                                                 dbappend=False,
                                                 save_sim=False,
                                                 parallel='mpi')
                sampler.sample(self.repetitions)

            elif self.calib_algorithm_streamflow == 'demcz':
                sampler = spotpy.algorithms.demcz(self,
                                                  dbname=dbname_dir,
                                                  dbformat="csv",
                                                  dbappend=False,
                                                  save_sim=False,
                                                  parallel='mpi')
                sampler.sample(self.repetitions)

            elif self.calib_algorithm_streamflow == 'dream':
                sampler = spotpy.algorithms.dream(self,
                                                  dbname=dbname_dir,
                                                  dbformat="csv",
                                                  dbappend=False,
                                                  save_sim=False,
                                                  parallel='mpi')
                sampler.sample(self.repetitions)
            elif self.calib_algorithm_streamflow == 'abc':
                sampler = spotpy.algorithms.abc(self,
                                                dbname=dbname_dir,
                                                dbformat="csv",
                                                dbappend=False,
                                                save_sim=False,
                                                parallel='mpi')
                sampler.sample(self.repetitions)
        elif self.set_calibrate == 0:
            name_ext_runoff = '_Runoff_ObjF_monthlyKGE'
            dbname_dir = (self.ModelPerformance +
                          self.calib_algorithm_runoff +
                          name_ext_runoff)
            if self.calib_algorithm_runoff == 'sceua':
                sampler = spotpy.algorithms.sceua(self,
                                                  dbname=dbname_dir,
                                                  dbformat="csv",
                                                  dbappend=False,
                                                  save_sim=False,
                                                  parallel='mpi')
                sampler.sample(self.repetitions, ngs=20,
                               kstop=20, peps=1e-7, pcento=1e-7)

            if self.calib_algorithm_runoff == 'NSGAII':
                n_pop = 40
                self.repetitions_nsgaii = int(self.repetitions / n_pop)
                sampler = spotpy.algorithms.NSGAII(self,
                                                   dbname=dbname_dir,
                                                   dbformat="csv",
                                                   dbappend=False,
                                                   save_sim=False,
                                                   parallel='mpi')
                sampler.sample(self.repetitions_nsgaii, n_obj=3, n_pop=n_pop)
            if self.calib_algorithm_runoff == 'demcz':
                sampler = spotpy.algorithms.demcz(self,
                                                  dbname=dbname_dir,
                                                  dbformat="csv",
                                                  dbappend=False,
                                                  save_sim=False,
                                                  parallel='mpi')
                sampler.sample(self.repetitions)
        # run with optimal parameters for final output
        optimal_params = bestParams_combination(self, dbname_dir + ".csv")
        self.calibration_run(optimal_params)

    def calibration_run(self, x):
        self.simulation(x)
        basin_ids_all = np.where(self.basin_ids == self.basin_num)[0]
        # if self.set_calibrate == -1:
        #     qsimulated = np.nanmean(
        #                     self.Avg_ChFlow[basin_ids_all, :], 0)
        if self.set_calibrate == 0:
            qsimulated = np.nansum(
                self.rsim[:, basin_ids_all] * self.conversion, 1
                )

        elif ~(self.set_calibrate == 0):
            qsimulated = self.Avg_ChFlow[
                        self.grdc_xanthosID, :]
        # sim
        if self.set_calibrate == -1:
            # KGE of the calibration period
            kge_cal = spotpy.objectivefunctions.kge(
                      self.bsn_obs[0:self.calib_length],
                      qsimulated[0:self.calib_length])
            # KGE of the validation period
            kge_val = spotpy.objectivefunctions.kge(
                       self.bsn_obs[self.calib_length + 1: self.data_length],
                       qsimulated[self.calib_length + 1:self.data_length])
            print(
                "Calibration KGE: {}, Validation KGE: {}".
                format(kge_cal, kge_val))
            # output
            print(
                "Simulated runoff output:" + self.out_dir +
                '/Basin_runoff_mmpermonth_Basin' +
                str(self.basin_num) + '.npy')
            print("Simulated streamflow output:" + self.out_dir +
                  '/Basin_Streamflow_m3persec_Basin' +
                  str(self.basin_num) + '.npy')
            out_streamflow = self.Avg_ChFlow[basin_ids_all, :]
            out_runoff = self.runoff[basin_ids_all, :]
            np.save(
                   self.out_dir +
                   '/Basin_Streamflow_m3persec_Basin' +
                   str(self.basin_num) + '.npy', out_streamflow)
            np.save(
                self.out_dir +
                '/Basin_runoff_mmpermonth_Basin' +
                str(self.basin_num) + '.npy', out_runoff)

        elif self.set_calibrate == 0:
            # KGE of the calibration period
            kge_cal = spotpy.objectivefunctions.kge(
                      self.bsn_obs_runoff[0:self.calib_length],
                      qsimulated[0:self.calib_length])
            # KGE of the validation period
            kge_val = spotpy.objectivefunctions.kge(
                      self.bsn_obs_runoff[
                        self.calib_length + 1:self.data_length],
                      qsimulated[self.calib_length + 1:self.data_length])
            print(
                "Calibration KGE: {}, Validation KGE: {}".
                format(kge_cal, kge_val))
            # output
            print(
                "Simulated runoff output:" +
                self.out_dir + '/Basin_runoff_mmpermonth_Basin' +
                str(self.basin_num) + '.npy')
            out_runoff = self.rsim.transpose()
            np.save(self.out_dir +
                    '/Basin_runoff_mmpermonth_Basin' +
                    str(self.basin_num) + '.npy', out_runoff)

        else:
            # KGE of the calibration period
            kge_cal = spotpy.objectivefunctions.kge(
                       self.bsn_obs[0:self.calib_length],
                       qsimulated[0:self.calib_length])
            # KGE of the validation period
            kge_val = spotpy.objectivefunctions.kge(
                      self.bsn_obs[self.calib_length + 1:self.data_length],
                      qsimulated[self.calib_length + 1:self.data_length])
            print(
                "Calibration KGE: {}, Validation KGE: {}".
                format(kge_cal, kge_val))
            # output
            print(
                "Simulated runoff output:" +
                self.out_dir + '/Basin_runoff_mmpermonth_Basin' +
                str(self.basin_num) + '.npy')
            print(
                "Simulated streamflow output:" +
                self.out_dir + '/Basin_Streamflow_m3persec_Basin' +
                str(self.basin_num) + '.npy')
            out_streamflow = self.Avg_ChFlow[basin_ids_all, :]
            out_runoff = self.runoff[basin_ids_all, :]
            np.save(self.out_dir +
                    '/Basin_Streamflow_m3persec_Basin' +
                    str(self.basin_num) + '.npy', out_streamflow)
            np.save(self.out_dir +
                    '/Basin_runoff_mmpermonth_Basin' +
                    str(self.basin_num) + '.npy', out_runoff)


def process_basin(basin_num, config_obj, calibration_data,
                  pet, router_function=None):
    """Process single basin."""

    # load ABCD runoff module data
    data_abcd = DataAbcd(config=config_obj)

    cal = CalibrateManaged(config_obj=config_obj,
                           basin_num=basin_num,
                           set_calibrate=config_obj.set_calibrate,
                           obs_unit=config_obj.obs_unit,
                           basin_ids=calibration_data.basin_ids,
                           basin_areas=calibration_data.area,
                           precip=data_abcd.precip,
                           pet=pet,
                           obs=calibration_data.cal_obs,
                           tmin=data_abcd.tmin,
                           runoff_spinup=config_obj.runoff_spinup,
                           router_func=router_function,
                           out_dir=config_obj.calib_out_dir,
                           )
    cal.calibrate_basin()


def calibrate_all(settings, calibration_data, pet, router_fn=None):
    """Run calibration for ABCD model for a basins."""
    import time

    if settings.n_basins == 235:
        for basin_num in expand_str_range(settings.cal_basins):
            # basin_name = calibration_data.basin_names[basin_num - 1]
            print("\tCalibrating Basin:  {}".format(basin_num))
            process_basin(basin_num, settings, calibration_data,
                          pet, router_function=router_fn)
            time.sleep(30)
    elif settings.n_basins == 1:
        basin_num = int(settings.basins_toRun)
        print("\tCalibrating Basin:  {}".format(basin_num))
        process_basin(basin_num, settings, calibration_data,
                      pet, router_function=router_fn)
    else:
        for basin_num in settings.basins_toRun:
            # basin_name = calibration_data.basin_names[basin_num - 1]
            print("\tCalibrating Basin:  {}".format(basin_num))
            process_basin(int(basin_num), settings, calibration_data,
                          pet, router_function=router_fn)
            time.sleep(30)


def plot_kge(calibration_result_file,
             output_file_name, dpi=300, figsize=(9, 5)):
    """Plot the KGE result of a calibrated basin"""

    split_extension = os.path.splitext(calibration_result_file)
    if len(split_extension) > 1:
        f = split_extension[0]
    else:
        f = calibration_result_file

    # load results
    results = spotpy.analyser.load_csv_results(f)

    # create plot
    fig = plt.figure(1, figsize=figsize)
    plt.plot(results['like1'] * -1)
    plt.ylabel('KGE')
    plt.xlabel('Repetition')
    fig.savefig(output_file_name, dpi=dpi)

    return plt


def bestParams_combination(self, performance_file):
    """Read in HDF data"""

    results = pd.read_csv(performance_file).dropna(axis=0)
    # sort parameter sets based on the objective function
    results_sorted = results.sort_values(
                     by='like1', ascending=False).reset_index(drop=True)

    if self.set_calibrate == 0:
        params_final = results_sorted.drop(['like1',
                                            'like2',
                                            'like3', 'chain'], 1)
    else:
        params_final = results_sorted[['para', 'parb',	'parc',
                                       'pard',	'parm', 'parwmbeta',
                                       'parwmalpha']]
    # select the best parameter set
    ro_params_selected = np.array(params_final.loc[0])

    return ro_params_selected

# converts monthly to annual


def timeseries_coverter(data_array, start_yr, ending_yr):
    sdate = date(int(start_yr), 1, 1)
    edate = date(int(ending_yr), 12, 31)
    data_ts = pd.DataFrame(data_array)
    data_ts.index = pd.date_range(start=sdate, end=edate, freq='M')
    mean_annual_data = np.squeeze(np.array(data_ts.resample('A').sum()))

    return mean_annual_data


def expand_str_range(str_ranges):
    """
    Expand a list of string ranges into full list of integers.

    Given a list of strings of integers or hyphen-separated integer ranges,
    expand the values to include the complete range. For example, if str_ranges
    is ['0-2', '6', '7-9'], this function will return [0, 1, 2, 6, 7, 8, 9].

    :param str_ranges:      List of strings, representing integer ranges
    """
    out_list = []
    for r in str_ranges:
        if '-' in r:
            start, end = r.split('-')
            out_list.extend(range(int(start), int(end) + 1))
        else:
            out_list.append(int(r))

    return out_list


def forcing_data_lakes(forcing_dict, nmth, nday, nm):
    ''' creates forcing data dictionary for month nm'''
    forcing_dict = {
                    'u_h': forcing_dict['wind_'][:, nm],
                    'ta_': forcing_dict['tas_'][:, nm],
                    'srad': forcing_dict['rsds_'][:, nm],
                    'lrad': forcing_dict['rlds_'][:, nm],
                    'rhs': forcing_dict['rhs_'][:, nm],
                    '_prcp': forcing_dict['prcp_'][:, nm],
                    'nmonth': nmth,
                    'nday': nday
                            }
    return forcing_dict
