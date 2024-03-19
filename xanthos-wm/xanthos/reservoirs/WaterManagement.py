"""
Water Management Components
@date: 08/22/2021
@authors: Guta Wakbulcho Abeshu : gwabeshu@uh.edu
          University of Houston
          HongYi Li : hli57@uh.edu
          University of Houston
"""
import sys
from datetime import date
import numpy as np
import pandas as pd


class Reservoir:
    ''' Reservoir release estimation
        Simulates three types of reservoirs :
                Irrigation, Hydropower and Flood Control
    '''
    def __init__(self, alpha, us_grids):
        """
        Initializes the Reservoir object.
        Parameters:
        alpha (float): The reservoir capacity scaling factor.
        grid_data (object): The data for the grids in upstream of dam.
        """
        self.alpha = alpha
        self.us_grids = us_grids

    def compute_channel_water_balance_error(self, qin, qout, sin, sout, dt_):
        """
        Computes channel water balance error,
        acceptable error is <=np.finfo(np.float32).eps
        """
        snorm = 0.5 * (sout + sin)
        dsdt_q = (qin - qout) * dt_
        dsdt_s = sout - sin
        ds_diff = np.abs(dsdt_q - dsdt_s)
        wbalance_relative_error = np.divide(ds_diff, snorm)
        wbalance_relative_error[snorm < 1] = 0

        if np.max(wbalance_relative_error) > np.finfo(np.float32).eps:
            us_grids_pd = np.squeeze(np.array(pd.DataFrame(self.us_grids)))
            _mn = wbalance_relative_error == np.max(wbalance_relative_error)

            print(f"Error: Channel water balance violated \
                        at XanthosID: {us_grids_pd[_mn]}")
            print(f"Qin: {qin[_mn]}")
            print(f"Qout: {qout[_mn]}")
            print(f"Sin: {sin[_mn]}")
            print(f"Sout: {sout[_mn]}")
            print(f"Error: {wbalance_relative_error[_mn]}")

            sys.exit()

        return np.max(wbalance_relative_error)

    # Reservoir water balance error
    def compute_reservoir_water_balance_error(self, qin, qout, sin, sout, dt_):
        """
        Computes reservoir water balance error,
        acceptable error is <=np.finfo(np.float32).eps
        """
        snorm = 0.5 * (sout + sin)
        dsdt_q = (qin - qout) * dt_
        dsdt_s = sout - sin
        ds_diff = np.abs(dsdt_q - dsdt_s)
        res_wbalance_relative_error = np.divide(ds_diff, snorm)
        res_wbalance_relative_error[snorm < 1] = 0

        if np.max(res_wbalance_relative_error) > np.finfo(np.float32).eps:
            us_grids_pd = np.squeeze(np.array(pd.DataFrame(self.us_grids)))
            _mm = np.where(
                res_wbalance_relative_error == np.max(
                    res_wbalance_relative_error))[0]
            print(f"Error: Reservoir water balance violated  \
                        at XanthosID: {us_grids_pd[_mm]}")
            print(f"Qin: {qin[_mm]}")
            print(f"Qout: {qout[_mm]}")
            print(f"Sin: {sin[_mm]}")
            print(f"Sout: {sout[_mm]}")
            print(f"Error: {np.max(res_wbalance_relative_error)}")

            sys.exit()

        return np.max(res_wbalance_relative_error)

    # #####################################################################
    # WATER MANAGEMENT
    # Irrigation Release
    def compute_irrigation_res_release(self, cpa, cond_ppose, qin, sini,
                                       mtifl, wdirr, irrmean):
        """
        Computes release from irrigation reservoirs.
        """
        nx_ = len(cond_ppose)
        rprovisional = np.zeros([nx_, ])  # Provisional Release
        rirrg_final = np.zeros([nx_, ])  # Final Release

        # Water management
        monthly_demand = wdirr[cond_ppose]  # Downstream demand: m^3/s
        mean_demand = irrmean[cond_ppose]  # Downstream mean demand:  m^3/s
        mtifl_irr = mtifl[cond_ppose]  # Mean flow:  m^3/s
        cpa_irr = cpa[cond_ppose]  # Capacity: 1e6 m^3
        qin_irr = qin[cond_ppose]
        sbegnining_of_year = sini[cond_ppose]

        # Provisional Release
        _m = mean_demand - (0.5 * mtifl_irr)
        cond1 = np.where(_m >= 0)[0]  # m >=0 ==> dmean >= 0.5*annual mean flow
        cond2 = np.where(_m < 0)[0]  # m < 0 ==> dmean < 0.5*annual mean inflow

        # Provisional Release
        demand_ratio = np.divide(monthly_demand[cond1], mean_demand[cond1])
        rprovisional[cond1] = np.multiply(
                0.5 * mtifl_irr[cond1], (1 + demand_ratio)
                )
        rprovisional[cond2] = (
                mtifl_irr[cond2] + monthly_demand[cond2] - mean_demand[cond2])

        # Final Release
        _c = np.divide(cpa_irr, (mtifl_irr * 365 * 24 * 3600))
        cond3 = np.where(_c >= 0.5)[0]  # c = capacity/imean >= 0.5
        cond4 = np.where(_c < 0.5)[0]  # c = capacity/imean < 0.5

        krls = np.divide(sbegnining_of_year, (self.alpha * cpa_irr))
        rirrg_final[cond3] = np.multiply(krls[cond3], mtifl_irr[cond3])

        temp1 = (_c[cond4] / 0.5) ** 2
        temp2 = np.multiply(temp1, krls[cond4])
        temp3 = np.multiply(temp2, rprovisional[cond4])
        temp4 = np.multiply((1 - temp1), qin_irr[cond4])
        rirrg_final[cond4] = temp3 + temp4

        return rirrg_final

    # Flood Control Release
    def compute_flood_control_res_release(self, cpa, cond_ppose,
                                          qin, sini, mtifl):
        """
        Computes release from flood control reservoirs.
        """
        nx_ = len(cond_ppose)
        rprovisional = np.zeros([nx_, ])  # Provisional Release
        rflood_final = np.zeros([nx_, ])  # Final Release

        # Water management
        mtifl_flood = mtifl[cond_ppose]  # Mean flow:  m^3/s
        cpa_flood = cpa[cond_ppose]  # Capacity:   m^3
        qin_flood = qin[cond_ppose]  # Mean flow:  m^3/s
        sbegnining_of_year = sini[cond_ppose]  # Capacity:   m^3

        # Provisional Release
        rprovisional = mtifl_flood.copy()

        # Final Release
        _c = np.divide(cpa_flood, (mtifl_flood * 365 * 24 * 3600))
        cond1 = np.where(_c >= 0.5)[0]  # c = capacity/imean >= 0.5
        cond2 = np.where(_c < 0.5)[0]  # c = capacity/imean < 0.5

        krls = np.divide(sbegnining_of_year, (self.alpha * cpa_flood))
        rflood_final[cond1] = np.multiply(krls[cond1], mtifl_flood[cond1])

        temp1 = (_c[cond2] / 0.5) ** 2
        temp2 = np.multiply(temp1, krls[cond2])
        temp3 = np.multiply(temp2, rprovisional[cond2])
        temp4 = np.multiply((1 - temp1), qin_flood[cond2])
        rflood_final[cond2] = temp3 + temp4

        return rflood_final

# Hydropower
class ReservoirTools:
    """
    Computes storage characteristics: levels, adjusting capacity_live based on each discretization.

    Parameters:
    - max_depth: Maximum depth of the reservoir.
    - head: Hydraulic head, m.
    - surface_area: Surface area of the reservoir at full capacity, km2.
    - capacity: Total reservoir storage capacity in Mm³.
    - capacity_live: Initial live capacity of the reservoir before adjustment, Mm³.
    - V: Array of water volume from 1000 reservoir discretizations, Mm³.

    Returns:
    - yh_: Computed water level in the reservoir, in m.
    - y_const: Constant derived from head and yh_, in m.
    - capacity_live_adjusted: Array of adjusted usable water volumes for each discretization, Mm³.
    """    
    @staticmethod
    def calculate_c(surface_area, capacity, max_depth=None):
        if max_depth is None:
            c = np.sqrt(2) / 3 * (surface_area * 10 ** 6) ** (3/2) / (capacity * 10 ** 6)
        else:
            c = 2 * capacity / (max_depth * surface_area)
        return c    

    @staticmethod        
    def get_level(c, V, max_depth, capacity):
        if max_depth is None:
            y = (6 * V / (c ** 2)) ** (1 / 3)
        else:
            y = max_depth * (V / (capacity * 10 ** 6)) ** (c / 2)
        return y    
    
    @staticmethod   
    def adjust_capacity(capacity_live, capacity, yconst, c, max_depth):
        if yconst < 0:
            if max_depth is None or np.isnan(max_depth):
                capacity_live = min(capacity_live, capacity - (-yconst) ** 3 * c ** 2 / 6 / 10 ** 6)
            else:
                capacity_live = min(capacity_live, capacity - (-yconst / max_depth) ** (2 / c) * capacity)
        return capacity_live


    @staticmethod
    def compute_storage_characteristics(max_depth, head, surface_area, capacity, capacity_live, V):
        c = ReservoirPolicy.calculate_c(surface_area, capacity, max_depth)
        
        V_m3 = V * 1e6  # Convert V from Mm³ to m³
        yh_ = ReservoirPolicy.get_level(c, V_m3, max_depth, capacity)
        y_const = head - yh_
        
        capacity_live_adjusted = np.zeros_like(V)
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                for k in range(V.shape[2]):
                    capacity_live_adjusted[i, j, k] = ReservoirPolicy.adjust_capacity(capacity_live, capacity, y_const[i,j,k], c, max_depth)


        return yh_, y_const


# Reservoir water balance
def reservoir_water_balance(Qin, Qout, Sin, cpa,
                            mtifl, alpha, dt,  resrvoirs_indx):
    """Re-adjusts release for environmental flow (if necessary) and
          Computes the storage level after release
          for all types of reservoirs"""
    # inputs
    Qin_ = Qin[resrvoirs_indx]
    Qout_ = Qout[resrvoirs_indx]
    Sin_ = Sin[resrvoirs_indx]
    cpa_ = cpa[resrvoirs_indx]
    mtifl_ = mtifl[resrvoirs_indx]

    # final storage and release initialization
    Nx = len(resrvoirs_indx)
    # environmental flow
    diff_rt = Qout_ - (mtifl_ * 0.1)
    indx_rt = np.where(diff_rt < 0)[0]
    Qout_[indx_rt] = 0.1 * mtifl_[indx_rt]

    # storage
    dsdt_resv = (Qin_ - Qout_) * dt
    Stemp = Sin_ + dsdt_resv

    Rfinal = np.zeros([Nx, ])    # final release
    Sfinal = np.zeros([Nx, ])    # final storage
    # condition a : storage > capacity
    Sa = (Stemp > (alpha * cpa_))
    if Sa.any():
        Sfinal[Sa] = alpha * cpa_[Sa]
        Rspill = (Stemp[Sa] - (alpha * cpa_[Sa])) / dt
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


def release_functions(res, dataflow, res_data, alpha):
    # factors
    secs_in_month = 2629800  # number of seconds in an average month
    cumecs_to_Mm3permonth = 2.6298  # m3/s to Mm3/month
    sww = 9810  # specific weight of water (N/m^3)
    # hours_in_year = 8766  # number of hours in a year
    # mwh_to_exajoule = 3.6 * (10 ** -9)  # megawatts to exajoules
    mths = 12  # months in year
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
    start_date = date(1971, 1, 1)  # Get start date for simulation "M/YYYY"
    Q = pd.DataFrame({'Q': inflow}, index=pd.date_range(start=start_date, periods=len(inflow), freq='ME'))         
    # array setup
    start_date = date(1971, 1, 1)  # Get start date for simulation "M/YYYY"
    Q_month_mat = inflow.set_index(pd.period_range(
                  start_date,
                  periods=len(inflow),
                  freq="M"))
    Q_month_mat = Q['Q'].groupby([Q.index.year, Q.index.month]).mean().unstack()
    Q_disc = np.array((0, 0.2375, 0.4750, 0.7125, 0.95, 1))
    Q_probs = np.diff(Q_disc)  # probabilities for each q class
    # Calculating adjusted probabilities
    adjusted_probs = Q_disc[:-1] + (Q_probs / 2)
    # Calculate the quantiles for each column at the adjusted probabilities
    Q_class_med = Q_month_mat.quantile(adjusted_probs, axis=0, interpolation='midpoint')
          
    # set up empty arrays to be populated
    shell_array = np.zeros(shape=(len(Q_probs), len(s_states), len(r_disc_x)))
    rev_to_go = np.zeros(len(s_states))
    # Bellman = np.zeros([len(s_states), m])
    r_policy_test = np.zeros([len(s_states), m])
    # work backwards through months of year (12 -> 1) and
    # repeat till policy converges

    while True:
        r_policy = np.zeros([len(s_states), num_months])
        for t in range(num_months, 0, -1):
            # constrained releases
            r_cstr = shell_array + q_class_med[t].values[np.newaxis, np.newaxis, :] + s_states[:, np.newaxis, np.newaxis]
            # desired releases
            r_star = shell_array + r_disc_x[np.newaxis, :, np.newaxis] 
            #r_star[:, 1:(r_disc + 1), :][r_star[:, 1:(r_disc + 1), :] > r_cstr[:, 1:(r_disc + 1), :]] = np.NaN

            s_nxt_stage = r_cstr - r_star
            s_nxt_stage[s_nxt_stage < 0] = 0
            s_nxt_stage[s_nxt_stage > capacity] = capacity
            s_avg = (s_nxt_stage + s_states[:, np.newaxis, np.newaxis]) /2

            # Compute storage level and head for all storage states
            yh_, y_const = ReservoirTools.compute_storage_characteristics(
                                        max_depth, 
                                        head, 
                                        surface_area, 
                                        capacity, 
                                        capacity,
                                        s_avg)
            h_arr = yh_ + y_const
            # ^^get head for all storage states for revenue calculation
            # revenue taken as head * release
            rev_arr = np.multiply(h_arr, r_star)
            implied_s_state = np.around(
                                1 + (s_nxt_stage / capacity) *
                                (len(s_states) - 1)).astype(np.int64)
            # ^^implied storage is the storage implied by each
            # release decision and inflow combination
            rev_to_go_arr = rev_to_go[implied_s_state - 1]
            max_rev_arr = rev_arr + rev_to_go_arr
            max_rev_arr_weighted = max_rev_arr * q_probs[np.newaxis, np.newaxis, :]
            # negative rev to reject non-feasible release
            max_rev_arr_weighted[r_star > r_cstr] = float("-inf")
            max_rev_expected = max_rev_arr_weighted.sum(axis=2) 
            rev_to_go = max_rev_expected.max(1)
            r_policy[:, t - 1] = np.argmax(max_rev_expected, 1)
        # Test for policy convergence
        pol_test = float(sum(sum(r_policy == r_policy_test))
                            ) / (num_months * len(s_states))
        r_policy_test = r_policy  # re-assign policy test for next loop test
        #print(pol_test)
        if pol_test >= tol:
            break

    return r_policy

    # Hydropower reservoir Release
    def compute_hydropower_res_release(self, inflow, release_policies,
                                       max_flow, capacity, initial_storage):
        """
        Computes the hydroelectric power reservoir release.
        """
        # Constants
        r_disc = 10
        s_disc = 1000
        cumecs_to_mm3permonth = 2.6298  # m3/s to Mm3/month

        # Convert to appropriate units
        capacity_mm3 = self.alpha * capacity * 1e-6  # MM3
        initial_storage_mm3 = initial_storage * 1e-6  # MM3

        num_reservoirs = len(capacity_mm3)
        computed_flow = np.zeros([num_reservoirs, ])
        final_storage = np.zeros([num_reservoirs, ])

        for reservoir_index in range(num_reservoirs):
            # Discretize storage and release states
            s_states = np.linspace(
                        0,
                        capacity_mm3[reservoir_index], s_disc + 1
                                  )
            r_disc_x = np.linspace(
                                   0,
                                   max_flow[reservoir_index
                                            ] * cumecs_to_mm3permonth,
                                   r_disc + 1)
            # Get current storage and inflow
            current_storage = initial_storage_mm3[reservoir_index]
            inflow_mm3 = inflow[reservoir_index] * cumecs_to_mm3permonth

            # Get release policy and compute release
            release_policy = release_policies[reservoir_index, :
                                              ].astype(np.int64)
            storage_diff = np.abs(s_states - current_storage)
            storage_state = np.where(storage_diff == np.min(storage_diff))
            release_index = release_policy[storage_state][0]
            release = r_disc_x[release_index]
            release = min(release, current_storage + inflow_mm3)
            recorded_release = release
            spill = 0

            # Compute next storage state
            dsdt_resv = inflow_mm3 - release
            temp_storage = current_storage + dsdt_resv
            if temp_storage > capacity_mm3[reservoir_index]:
                final_storage[reservoir_index] = capacity_mm3[reservoir_index]
                spill = temp_storage - capacity_mm3[reservoir_index]
            else:
                if temp_storage < 0:
                    final_storage[reservoir_index] = 0
                    recorded_release = max(0, current_storage + inflow_mm3)
                else:
                    final_storage[reservoir_index] = temp_storage

            # Convert Mm3/month to m3/s
            computed_flow[reservoir_index] = (recorded_release + spill
                                              ) / cumecs_to_mm3permonth

        return computed_flow

    # Reservoir water balance
    def compute_reservoir_water_balance(self, inflow, outflow, initial_storage,
                                        capacity, mean_flow, dt_,
                                        reservoir_indices):
        """
        Re-adjusts release for environmental flow (if necessary) and
        computes the storage level after release for all types of reservoirs.
        """
        # Extract reservoir-specific data
        inflow_ = inflow[reservoir_indices]
        outflow_ = outflow[reservoir_indices]
        initial_storage_ = initial_storage[reservoir_indices]
        capacity_ = capacity[reservoir_indices]
        mean_flow_ = mean_flow[reservoir_indices]

        # Initialize final storage and release arrays
        num_reservoirs = len(reservoir_indices)
        final_release = np.zeros([num_reservoirs, ])
        final_storage = np.zeros([num_reservoirs, ])

        # Adjust outflow for environmental flow
        outflow_diff = outflow_ - (mean_flow_ * 0.1)
        low_outflow_indices = np.where(outflow_diff < 0)[0]
        outflow_[low_outflow_indices] = 0.1 * mean_flow_[low_outflow_indices]

        # Compute change in reservoir storage
        dsdt_reservoir = (inflow_ - outflow_) * dt_
        temp_storage = initial_storage_ + dsdt_reservoir

        # Condition a: storage > capacity
        over_capacity_indices = (
                        temp_storage > (self.alpha * capacity_))
        if over_capacity_indices.any():
            final_storage[over_capacity_indices
                          ] = self.alpha * capacity_[over_capacity_indices]
            spill_release = (temp_storage[over_capacity_indices] -
                             (self.alpha * capacity_[over_capacity_indices])
                             ) / dt_
            final_release[over_capacity_indices
                          ] = outflow_[over_capacity_indices] + spill_release

        # Condition b: storage <= 0
        negative_storage_indices = temp_storage < 0
        if negative_storage_indices.any():
            final_storage[negative_storage_indices] = 0
            final_release[negative_storage_indices
                          ] = (initial_storage_[negative_storage_indices] / dt_
                               ) + inflow_[negative_storage_indices]

        # Condition c: 25% capacity < S < capacity
        within_capacity_indices = (
                (temp_storage > 0) & (temp_storage <= self.alpha * capacity_))
        if within_capacity_indices.any():
            final_storage[within_capacity_indices
                          ] = temp_storage[within_capacity_indices]
            final_release[within_capacity_indices
                          ] = outflow_[within_capacity_indices]

        return final_release, final_storage
