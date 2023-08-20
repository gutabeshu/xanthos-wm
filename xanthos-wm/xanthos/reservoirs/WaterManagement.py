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
    @staticmethod
    def compute_storage_level(max_depth, head, surface_area,
                              capac, cap_live, volume):
        """
        Computes storage level for hydropower reservoirs.
        """
        if np.isnan(max_depth):
            cp_ = (np.sqrt(2) / 3) * (
                (surface_area * 1e6) ** (3/2)) / (capac * 1e6)
            yh_ = (6 * volume / (cp_ ** 2)) ** (1 / 3)
            y_const = head - yh_
            if y_const < 0:
                cap_live = np.nanmin([
                    cap_live,
                    capac - (((-y_const) ** 3) * (cp_ ** 2 / 6 / 1e6))
                ])
        else:
            cp_ = 2 * capac / (max_depth * surface_area)
            yh_ = max_depth * (volume / (capac * 1e6)) ** (cp_ / 2)
            y_const = head - max_depth
            if y_const < 0:
                cap_live = np.nanmin([
                    cap_live,
                    capac - ((-y_const / max_depth) ** (2 / cp_) * capac)
                ])

        return yh_, y_const, cap_live

    # Hydropower reservoirs policy
    def compute_release_policies(self, reservoir_id, dataflow, reservoir_data):
        """
        Computes release policies for reservoirs.
        """
        # Constants
        secs_in_month = 2629800  # Number of seconds in an average month
        cumecs_to_mm3permonth = 2.6298  # m3/s to Mm3/month
        specific_weight_water = 9810  # Specific weight of water (N/m^3)
        num_months = 12  # Months in a year

        # Reservoir specific data
        inflow = dataflow[reservoir_id, :]
        cap = self.alpha * reservoir_data["CAP"][reservoir_id]  # MCM
        cap_live = self.alpha * reservoir_data["CAP"][reservoir_id]  # MCM
        installed_capacity = reservoir_data["ECAP"][reservoir_id]
        q_max = reservoir_data[
                        "FLOW_M3S"][reservoir_id] * cumecs_to_mm3permonth
        efficiency = 0.9
        surface_area = reservoir_data["AREA"][reservoir_id]
        max_depth = reservoir_data["DAM_HGT"][reservoir_id]
        head = max_depth

        if np.isnan(head):
            head = installed_capacity / (
                efficiency * specific_weight_water * (q_max / secs_in_month))

        if np.isnan(q_max):
            q_max = (installed_capacity / (
                    efficiency * specific_weight_water * head)) * secs_in_month

        # Setup for stochastic dynamic programming
        r_disc = 10
        s_disc = 1000
        s_states = np.linspace(0, cap, s_disc + 1)
        r_disc_x = np.linspace(0, q_max, r_disc + 1)
        inflow_mm3 = inflow * cumecs_to_mm3permonth

        # Array setup
        start_date = date(1971, 1, 1)  # Start date for simulation "M/YYYY"
        inflow_dataframe = pd.DataFrame(inflow_mm3)
        inflow_by_month = inflow_dataframe.set_index(
                pd.period_range(start_date,
                                periods=len(inflow_dataframe), freq="M"))

        # Quantiles for inflow classes
        q_disc = np.array((0, 0.2375, 0.4750, 0.7125, 0.95, 1))
        q_probs = np.diff(q_disc)  # Probabilities for each q class
        q_class_med = inflow_by_month.groupby(
                        inflow_by_month.index.month).quantile(
                            q_disc[1:6] - (q_probs / 2))

        # Setup arrays to be populated
        shell_array = np.zeros(shape=(
                        len(q_probs), len(s_states), len(r_disc_x)))
        rev_to_go = np.zeros(len(s_states))
        r_policy_test = np.zeros([len(s_states), num_months])

        # Work backwards through months of year (12 -> 1) and
        # repeat till policy converges
        while True:
            r_policy = np.zeros([len(s_states), num_months])

            for _t in range(num_months, 0, -1):
                # Compute constrained releases and desired releases
                r_cstr = shell_array + np.array(
                            q_class_med.loc[_t, slice(None)][0]
                        )[:, None][:, None] + shell_array + s_states[:, None]
                r_star = shell_array + r_disc_x

                # Compute next stage storage
                s_nxt_stage = r_cstr - r_star
                s_nxt_stage[s_nxt_stage < 0] = 0
                s_nxt_stage[s_nxt_stage > cap] = cap

                # Compute storage level and head for all storage states
                yh_, y_const, cap_live = self.compute_storage_level(
                                max_depth, head, surface_area, cap, cap_live,
                                (s_nxt_stage + s_states[:, None]) * 1e6 / 2)
                h_arr = yh_ + y_const

                # Compute revenue taken as head * release
                rev_arr = np.multiply(h_arr, r_star)

                # Compute implied storage state
                implied_s_state = np.around(
                                  1 + (s_nxt_stage / cap) * (
                                    len(s_states) - 1)).astype(np.int64)
                # Compute expected maximum revenue
                rev_to_go_arr = rev_to_go[implied_s_state - 1]
                max_rev_arr = rev_arr + rev_to_go_arr
                max_rev_arr_weighted = max_rev_arr * np.array(
                                                q_probs)[:, None][:, None]
                # Negative revenue to reject non-feasible release
                max_rev_arr_weighted[r_star > r_cstr] = float("-inf")

                # Update revenue to go
                max_rev_expected = max_rev_arr_weighted.sum(axis=0)
                rev_to_go = max_rev_expected.max(1)

                # Update policy
                r_policy[:, _t - 1] = np.argmax(max_rev_expected, 1)

            # Test for policy convergence
            pol_test = float(sum(sum(r_policy == r_policy_test))
                             ) / (num_months * len(s_states))
            r_policy_test = r_policy  # Re-assign policy test for next loop

            if pol_test >= 0.99:
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
