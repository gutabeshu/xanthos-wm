"""
Calculate country and GCAM-region level hydropower production from Xanthos streamflow.

Created on April 25, 2017

@author: Sean Turner (sean.turner@pnnl.gov)
@Project: Xanthos V2.0

Modified on April 7, 2023
@author : Guta W. Abeshu (gwabeshu@uh.edu)
@change : rule curve optimization , previously an input

License:  BSD 2-Clause, see LICENSE and DISCLAIMER files

Copyright (c) 2017, Battelle Memorial Institute
"""

import math
import numpy as np
import pandas as pd
import os
from datetime import date


class HydropowerActual:
    """
    Compute country and GCAM-region level hydropower production time series based on xanthos streamflow output.

    Gridded streamflow is used to drive dam simulations for 1593 large dams (~54% global hydropower installed capacity).
    Each dam is has been pre-assigned an optimized look-up table, which assigns a turbine release at each time step
    using current reservoir storage, inflow and month of year. Power output time series are summed across dams for each
    country and scaled up to account for unsimulated capacity (i.e., for dams not represented in the model).
    """

    secs_in_month = 2629800  # number of seconds in an average month
    cumecs_to_Mm3permonth = 2.6298  # m3/s to Mm3/month
    sww = 9810  # specific weight of water (N/m^3)
    hours_in_year = 8766  # number of hours in a year
    hours_in_month = 30.4375*24 # number of hours in a month
    mwh_to_exajoule = 3.6 * (10 ** -9)  # megawatts to exajoules

    def __init__(self, settings, q_grids):
        """Load inputs, run simulation, and output results."""
        self.settings = settings
        self.q_grids = q_grids
        # read input data
        self.res_data = pd.read_csv(settings.HydroDamData)  # Read dam data
        self.grid_data = pd.read_csv(settings.GridData)  # Read grid data
        self.drainage_area = np.loadtxt(settings.DrainArea)  # Read upstream drainage area for all grids
        self.missing_cap = pd.read_csv(settings.MissingCap)  # Read missing installed capacity (i.e., dams not represented)

        # read input data
        self.res_data = pd.read_csv(self.HydroDamData)  # Read dam data
        self.grid_data = pd.read_csv(self.GridData)  # Read grid data
        self.drainage_area = np.loadtxt(self.DrainArea)  # Read upstream drainage area for all grids
        self.missing_cap = pd.read_csv(self.MissingCap)  # Read missing installed capacity (i.e., dams not represented)

        # assign from inputs
        self.fname_by_region = "actual_hydro_by_gcam_region_EJperyr_{}.csv".format(settings.ProjectName)
        self.fname_by_country = "actual_hydro_by_countries_EJperyr_{}.csv".format(settings.ProjectName)
        self.fname_by_dams = "actual_hydro_by_dam_EJpermth_{}.csv".format(settings.ProjectName)
        self.OutputFolder = settings.OutputFolder
        self.start_date = "1/1971" #settings.hact_start_date  # Get start date for simulation "M/YYYY"
        self.loc_refs = self.grid_data[["ID", "long", "lati"]]  # Get latitiude and longitude for all grid squares
        self.grid_ids = self.res_data.iloc[:, 0:2].apply(self.get_grid_id, 1)  # Get grid indices for all dams
        self.q = np.transpose(self.q_grids[self.grid_ids - 1, :])  # get flows for all grids containing dams
        self.rule_curves = np.zeros([self.res_data.shape[0], 1001, 12])  # rule curves for all 1593 dams
        for res in range(self.res_data.shape[0], 1001, 12):
            self.rule_curves[res, :, :] = self.get_rule_curve(res,
                                                              self.q,
                                                              self.res_data,
                                                              )

        self.dr_ar_assumed = np.apply_along_axis(self.get_drain_area,
                                                 1,
                                                 np.array(self.loc_refs)[self.grid_ids - 1, 1:3],
                                                 self.drainage_area)  # Get assumed drainage areas (grid)
        self.dr_ar_actual = np.array(self.res_data["CATCH"])  # Get actual drainage areas
        self.q_ = self.q * self.dr_ar_actual / self.dr_ar_assumed  # Correct inflow by drainage
        self.q_Mm3 = self.q_ * HydropowerActual.cumecs_to_Mm3permonth  # Get inflow in Mm^3 / month
        self.power_all_dams = np.empty([len(self.q_Mm3), len(self.res_data)])  # array to hold power data

        # assigned during run
        self.rc = None
        self.inflow = None
        self.cap = None
        self.cap_live = None
        self.head = None
        self.q_max = None
        self.efficiency = None
        self.l_inflow = None
        self.s = None
        self.power = None
        self.s_states = None
        self.q_states = None
        self.env_flow = None
        self.hydro_gcam_regions_EJ = None

        # run simulation
        self.hydro_sim()

        # convert power production time series to GCAM region enery production
        self.to_region()

        # write output
        self.write_output()

    @staticmethod
    def find_nearest(array, value):
        """Get value from within an array closest to a value."""
        idx = (np.abs(array - value)).idxmin()  # idxmin instead of argmin
        return array[idx]

    @staticmethod
    def find_nearest_idx(array, value):
        """Get index of an array closest to a value."""
        return (np.abs(array - value)).idxmin()

    def get_grid_id(self, longlat):
        """Get the grid location of a dam based on longitude/latitude."""
        lon = self.find_nearest(self.loc_refs["long"], longlat["LONG_DD"])
        lat = self.find_nearest(self.loc_refs["lati"], longlat["LAT_DD"])
        return int(self.loc_refs[(self.loc_refs["long"] == lon) & (self.loc_refs["lati"] == lat)]["ID"])

    def get_drain_area(self, x, drainage_area):
        """Get drainage area implied by the routing network."""
        lonseq = np.unique(self.loc_refs["long"])
        latseq = np.unique(self.loc_refs["lati"])[::-1]
        return drainage_area[latseq.tolist().index(x[1]), lonseq.tolist().index(x[0])]

    def env_flow_constraint(self):
        """Apply environmental flow constraints."""
        inflow_mmf = self.inflow.groupby(self.inflow.index.month).mean()
        inflow_maf = self.inflow.mean()
        inflow_efr_perc = ((inflow_mmf < 0.4 * inflow_maf) * 1 * 0.6 +
                           ((inflow_mmf >= 0.4 * inflow_maf) & (inflow_mmf <= 0.8 * inflow_maf)) * 1 * 0.45 +
                           (inflow_mmf > 0.8 * inflow_maf) * 1 * 0.3 * [not i for i in (inflow_mmf < 1)])
        self.env_flow = inflow_efr_perc * inflow_mmf

    def init_sim(self):
        """Initialize simulation."""
        self.s = [self.cap] * (self.l_inflow + 1)  # Declare storage variable to be simulated
        self.power = [0] * self.l_inflow
        self.q_states = self.inflow.groupby(self.inflow.index.month).quantile((0, 0.2375, 0.4750, 0.7125, 0.95, 1))

    def get_power(self, res):
        """
        MAIN SIMULATION FUNCTION.

        NEXT VERSION TO INCLUDE ev, area, max_depth, installed_cap
        """
        # start simulation (method for getting q state should be improved--currently inaccurate)
        for t in range(0, self.l_inflow):
            mth = self.inflow.index[t].month
            if t == 0:
                Sini = self.cap
            else:
                Sini = self.s[t-1]
            release = HydropowerActual.hp_release(self.inflow[t],
                                                  self.rc[:, mth-1],
                                                  self.q_max,
                                                  self.cap,
                                                  Sini)

            env = self.env_flow[mth]
            active = self.s[t] + self.inflow[t] - (self.cap - self.cap_live)
            r = min(min(max(release, env), active), self.q_max)
            self.s[t + 1] = max(min(self.s[t] + self.inflow[t] - r,
                                    self.cap), 0)
            h = (np.mean([self.s[t], self.s[t + 1]]) / self.cap) * self.head
            self.power[t] = max(self.efficiency * HydropowerActual.sww * h * (
                                r / HydropowerActual.secs_in_month), 0)

        self.power_all_dams[:, res] = self.power  # MegaWatts

    def GetLevel(max_depth, head, surface_area, capac, cap_live, V):
        """Computes storage level for hydropower reservoirs"""

        if np.isnan(max_depth):
            c = (np.sqrt(2) / 3) * ((surface_area * 1e6) ** (3/2)
                                    ) / (capac * 1e6)
            y = (6 * V / (c**2)) ** (1 / 3)
            yconst = head - y
            if (yconst < 0):
                cap_live = np.nanmin([
                        cap_live,
                        capac - (((-yconst) ** 3) * (c ** 2 / 6 / 1e6))])
        else:
            c = 2 * capac / (max_depth * surface_area)
            y = max_depth * (V / (capac * 1e6))**(c / 2)
            yconst = head - max_depth
            if (yconst < 0):
                cap_live = np.nanmin([
                        cap_live,
                        capac - (
                            (-yconst / max_depth) ** (2 / c) * capac)])

        return y, yconst, cap_live

    def get_rule_curve(self, res, dataflow, res_data):
        # factors
        secs_in_month = 2629800  # number of seconds in an average month
        cumecs_to_Mm3permonth = 2.6298  # m3/s to Mm3/month
        sww = 9810  # specific weight of water (N/m^3)
        # hours_in_year = 8766  # number of hours in a year
        # mwh_to_exajoule = 3.6 * (10 ** -9)  # megawatts to exajoules
        mths = 12  # months in year
        alpha = 0.85
        # inflow
        QQ = dataflow[res, :]
        # reserv
        # #######################
        cap = alpha*res_data["CAP"][res]  # MCM
        cap_live = alpha*res_data["CAP"][res]  # MCM
        installed_cap = res_data["ECAP"][res]
        q_max = res_data["FLOW_M3S"][res] * cumecs_to_Mm3permonth
        efficiency = 0.9
        surface_area = res_data["AREA"][res]
        max_depth = res_data["DAM_HGT"][res]
        head = max_depth
        if np.isnan(head):
            head = installed_cap / (efficiency * sww * (q_max / secs_in_month))
        if np.isnan(q_max):
            qmax = (installed_cap / (efficiency * sww * head)) * secs_in_month
        ######################
        # storage discretized to 1000 segments for stoch. dyn. prog.
        r_disc = 10
        s_disc = 1000
        s_states = np.linspace(0, cap, s_disc+1)
        r_disc_x = np.linspace(0, q_max, r_disc+1)
        m = mths
        q_Mm3 = QQ*cumecs_to_Mm3permonth
        inflow = pd.DataFrame(q_Mm3)
        # array setup
        start_date = date(1971, 1, 1)  # Get start date for simulation "M/YYYY"
        Q_month_mat = inflow.set_index(pd.period_range(
                    start_date,
                    periods=len(inflow),
                    freq="M"))
        Q_disc = np.array((0, 0.2375, 0.4750, 0.7125, 0.95, 1))
        Q_probs = np.diff(Q_disc)  # probabilities for each q class
        Q_class_med = Q_month_mat.groupby(
                    Q_month_mat.index.month).quantile(
                        Q_disc[1:6] - (Q_probs / 2))
        # set up empty arrays to be populated
        shell_array = np.zeros(shape=(len(Q_probs),
                                      len(s_states), len(r_disc_x)))
        rev_to_go = np.zeros(len(s_states))
        # Bellman = np.zeros([len(s_states), m])
        r_policy_test = np.zeros([len(s_states), m])
        # work backwards through months of year (12 -> 1) and
        # repeat till policy converges

        while True:
            r_policy = np.zeros([len(s_states), m])
            for t in range(m, 0, -1):
                # constrained releases
                r_cstr = shell_array + np.array(
                                    Q_class_med.loc[t, slice(None)][0]
                                    )[:, np.newaxis][:, np.newaxis] + \
                                        shell_array + s_states[:, np.newaxis]
                # desired releases
                r_star = shell_array + r_disc_x
                s_nxt_stage = r_cstr - r_star
                s_nxt_stage[s_nxt_stage < 0] = 0
                s_nxt_stage[s_nxt_stage > cap] = cap
                y, yconst, cap_live = self.GetLevel(
                                                    max_depth,
                                                    head,
                                                    surface_area,
                                                    cap,
                                                    cap_live,
                                                    (s_nxt_stage +
                                                     s_states[:, np.newaxis]
                                                     )*1e6 / 2)
                h_arr = y + yconst
                # ^^get head for all storage states for revenue calculation
                # revenue taken as head * release
                rev_arr = np.multiply(h_arr, r_star)
                implied_s_state = np.around(
                                1 + (s_nxt_stage / cap) *
                                (len(s_states) - 1)).astype(np.int64)
                # ^^implied storage is the storage implied by each
                # release decision and inflow combination
                rev_to_go_arr = rev_to_go[implied_s_state - 1]
                max_rev_arr = rev_arr + rev_to_go_arr
                max_rev_arr_weighted = max_rev_arr * np.array(
                                    Q_probs)[:, np.newaxis][:, np.newaxis]
                # negative rev to reject non-feasible release
                max_rev_arr_weighted[r_star > r_cstr] = float("-inf")
                max_rev_expected = max_rev_arr_weighted.sum(axis=0)
                rev_to_go = max_rev_expected.max(1)
                r_policy[:, t - 1] = np.argmax(max_rev_expected, 1)
            pol_test = float(
                            sum(sum(
                                r_policy == r_policy_test))
                                ) / (m * len(s_states))
            r_policy_test = r_policy  # re-assign policy test for next loop
            # print(pol_test)
            if pol_test >= 0.99:
                break

        return r_policy

    def hp_release(inflow_data, rpolicy_all, q_max, cap, Sinn):
        r_disc = 10
        s_disc = 1000
        alpha = 0.85
        cap = alpha*cap*1e6  # MM3
        Sin = Sinn*1e6  # MM3

        # storage discretized to 1000 segments for stoch. dyn. prog.
        s_states = np.linspace(0, cap, s_disc+1)
        r_disc_x = np.linspace(0, q_max, r_disc+1)
        Sct = Sin
        Qx = inflow_data
        r_policy = rpolicy_all.astype(np.int64)
        s_diff = np.abs(s_states - Sct)
        S_state = np.where(s_diff == np.min(s_diff))
        indxt = r_policy[S_state][0]
        R = r_disc_x[indxt]
        R = min(R, Sct + Qx)  # - (capacity - capacity_live))
        R_rec = R
        Spill = 0
        s_beggining = Sct
        dsdt_resv = (Qx-R)
        Stemp = s_beggining + dsdt_resv
        if Stemp > cap:
            Spill = Stemp - cap
        elif Stemp < 0:
            R_rec = max(0, s_beggining + Qx)

        # release in Mm3/month
        q_release = (R_rec + Spill)

        return q_release

    def sim_vars(self, idx, res):
        """Calculate simulation variable for a target reservoir."""
        # get rule curves
        self.rc = self.rule_curves[res, :, :]

        self.inflow = \
            pd.DataFrame(self.q_Mm3).set_index(pd.period_range(self.start_date, periods=len(self.q_Mm3), freq="M"))[res]
        self.l_inflow = len(self.inflow)
        self.cap = self.res_data["CAP"][res]
        self.cap_live = self.res_data["CAPLIVE"][res]

        if math.isnan(self.cap_live) is True:
            self.cap_live = self.cap

        self.installed_cap = self.res_data["ECAP"][res]
        self.q_max = self.res_data["FLOW_M3S"][res] * HydropowerActual.cumecs_to_Mm3permonth
        self.efficiency = self.res_data["EFF"][res]
        self.head = self.res_data["HEAD"][res]

        if math.isnan(self.head) is True:
            self.head = self.installed_cap / (
                self.efficiency * HydropowerActual.sww * (self.q_max / HydropowerActual.secs_in_month))

    def hydro_sim(self):
        """Simulate hydropower operations and add power to array."""
        for idx, res in enumerate(self.res_data.index.tolist()):
            # calculate sim variables
            self.sim_vars(idx, res)

            # environmental flow constraint
            self.env_flow_constraint()

            # initialize simulation
            self.init_sim()

            # get power
            self.get_power(res)

    def to_region(self):
        """Convert power production time series to GCAM region enery production."""
        self.power_all_dams_monthly_MW = pd.DataFrame(self.power_all_dams).set_index(
            pd.period_range(self.start_date, periods=len(self.power_all_dams), freq="M")
        )
        # MW to EJ
        self.power_all_dams_monthly_EJ = self.power_all_dams_monthly_MW * (
            HydropowerActual.hours_in_month * HydropowerActual.mwh_to_exajoule)
        #
        self.energy_all_dams = self.power_all_dams_monthly_MW.resample("A").mean() * (
            HydropowerActual.hours_in_year * HydropowerActual.mwh_to_exajoule
        )
        self.energy_all_countries = self.energy_all_dams.groupby(self.res_data["COUNTRY"], axis=1).sum()
        self.energy_all_countries_total = self.energy_all_countries.multiply(list(self.missing_cap["factor"]))
        self.hydro_gcam_regions_EJ = self.energy_all_countries_total.groupby(list(self.missing_cap["GCAM_ID"]), axis=1).sum()

    def write_output(self):
        """Write results to CSV."""
        # df = pd.DataFrame(self.hydro_gcam_regions_EJ)
        # cols = ['region_{}'.format(i) for i in df.columns]
        # cols.insert(0, 'year')
        # df.columns = cols
        # by region
        path = os.path.join(self.OutputFolder, self.fname_by_region)
        by_region_df = self.hydro_gcam_regions_EJ.T
        by_region_df.reset_index(inplace=True)
        by_region_df.rename(columns={'index': 'region'}, inplace=True)
        pd.DataFrame.to_csv(by_region_df, path, index=False)

        # by country
        path = os.path.join(self.OutputFolder, self.fname_by_country)
        by_region_df = self.energy_all_countries_total.T
        by_region_df.reset_index(inplace=True)
        by_country_df = pd.concat([self.missing_cap, by_region_df], 1)
        pd.DataFrame.to_csv(by_country_df, path, index=False)

        # by dams
        path = os.path.join(self.OutputFolder, self.fname_by_dams)
        by_reservoir_df = pd.concat([self.res_data,
                                     self.power_all_dams_monthly_EJ.T], 1)
        pd.DataFrame.to_csv(by_reservoir_df, path, index=False)
