import unittest

import numpy as np
import scipy.sparse as sparse

from xanthos.routing import mrtm_managed as routing


class TestMrtmManagedReservoirCoupling(unittest.TestCase):
    """Synthetic checks for reservoir release coupling into downstream routing."""

    def test_hydropower_effective_head_prefers_capacity_derived_head(self):
        head = routing._hydropower_effective_head(
            installed_cap=100.0,
            max_turbine_flow=50.0,
            dam_height=999.0,
        )

        expected = 100.0 * 1e6 / (9810 * 50.0 * 0.9)
        self.assertAlmostEqual(head, expected)

    def test_hydropower_effective_head_falls_back_to_dam_height(self):
        head = routing._hydropower_effective_head(
            installed_cap=100.0,
            max_turbine_flow=0.0,
            dam_height=75.0,
        )

        self.assertEqual(head, 75.0)

    def test_zero_release_reservoir_blocks_downstream_channel_inflow(self):
        # Two-cell network: cell 0 flows into cell 1.
        ncell = 2
        up = sparse.coo_matrix(
            (np.array([1.0]), (np.array([1]), np.array([0]))),
            shape=(ncell, ncell)
        )
        um = up - sparse.eye(ncell, dtype=float)

        flow_distance = np.array([100000.0, 100000.0])
        channel_velocity = np.array([1.0, 1.0])
        channel_storage = np.array([1000.0, 0.0])
        instream_flow = np.zeros(ncell)
        runoff = np.zeros(ncell)
        area = np.ones(ncell)
        nday = 1
        dt = 43200
        initial_year_storage = np.array([5e8, 0.0])
        previous_reservoir_storage = np.array([5e8, 0.0])
        demand = np.zeros(ncell)
        mean_demand = np.zeros(ncell)
        mean_inflow = np.zeros(ncell)
        purpose = np.array([1, 0])
        capacity = np.array([1e9, 0.0])
        release_policy = np.zeros((1, 1001))
        max_turbine_flow = np.zeros(ncell)
        water_consumption = np.zeros(ncell)
        alpha = 0.85
        active_grids = np.array([0, 1])

        result = routing.streamrouting(
            flow_distance,
            channel_storage,
            instream_flow,
            channel_velocity,
            runoff,
            area,
            nday,
            dt,
            um,
            up,
            initial_year_storage,
            demand,
            mean_demand,
            mean_inflow,
            purpose,
            capacity,
            release_policy,
            max_turbine_flow,
            water_consumption,
            alpha,
            previous_reservoir_storage,
            1,
            active_grids
        )

        avg_channel_flow = result[1]
        final_channel_storage = result[0]
        avg_channel_inflow = result[3]
        avg_reservoir_inflow = result[5]
        avg_reservoir_release = result[6]

        self.assertAlmostEqual(avg_channel_flow[0], 0.0)
        self.assertAlmostEqual(avg_channel_flow[1], 0.0)
        self.assertAlmostEqual(final_channel_storage[1], 0.0)
        self.assertAlmostEqual(avg_channel_inflow[1], 0.0)
        self.assertGreater(avg_reservoir_inflow[0], 0.0)
        self.assertAlmostEqual(avg_reservoir_release[0], 0.0)


if __name__ == '__main__':
    unittest.main()
