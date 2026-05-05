import xarray
import numpy as np
import pandas as pd
from xanthos.data_reader.data_reference import DataReference


class DataCalibrationManaged(DataReference):
    """Load data for calibration that uses streamflow and
         accounts for water management."""

    def __init__(self,
                 config_obj=None,
                 cal_observed=None,
                 optimal_parameters_file=None,
                 purpose_file=None,
                 capacity_file=None,
                 hp_release_file=None,
                 water_consumption_file=None,
                 instream_flownat_file=None,
                 initial_chs_nat_file=None,
                 sm_file=None,
                 mtif_natural_file=None,
                 maxtif_natural_file=None,
                 total_demand_cumecs_file=None,
                 grdc_coord_index_file=None,
                 start_year=None,
                 end_year=None):

        if config_obj is None:

            self.start_year = start_year
            self.end_year = end_year
            self.nmonths = (self.end_year - self.start_year + 1) * 12

            super().__init__(nmonths=self.nmonths)

            # use basin-level flow as target for calibration; select only columns for basin number and runoff
            try:
                self.cal_obs = self.load_data(cal_observed, 0)[:, [0, 3]]
            except AttributeError:
                pass

            # load dam and other input data
            # self.purpose = np.load(purpose_file)
            # self.capacity = np.load(capacity_file)
            # self.hp_release = np.load(hp_release_file)
            # self.water_consumption = np.load(water_consumption_file)
            # self.instream_flow_natural = np.load(instream_flow_natural_file)
            # self.ini_channel_storage = np.load(
            # initial_channel_storage_natural_file)
            # self.sm = np.load(sm_file)
            # self.mtif_natural = np.load(mtif_natural_file)
            # self.maxtif_natural = np.load(maxtif_natural_file)
            # self.total_demand_cumecs = np.load(total_demand_cumecs_file)
            # self.grdc_coord_index_file = np.load(grdc_coord_index_file)
        else:

            self.config_obj = config_obj
            self.start_year = config_obj.StartYear
            self.end_year = config_obj.EndYear
            self.nmonths = config_obj.nmonths
            self.set_calibrate = config_obj.set_calibrate

            super().__init__(nmonths=self.nmonths)

            # use basin-level flow as target for calibration;
            # select only columns for basin number and runoff
            try:
                if self.set_calibrate == 0:
                    self.cal_obs = self.load_data(self.config_obj.cal_observed, 0)
                else:
                    self.cal_obs = self.load_data(self.config_obj.cal_observed, 0)[:, [0, 3]]

            except AttributeError:
                pass

            # load xanthos wm file
            Xanthos_wm = xarray.open_dataset(self.config_obj.Xanthos_wm_file)

            def one_dim(name, default=0.0, keep_nan=False):
                if name in Xanthos_wm:
                    values = Xanthos_wm[name].values
                    return values if keep_nan else np.nan_to_num(values)
                return np.full(self.ncells, default, dtype=float)

            def monthly(name):
                return np.nan_to_num(Xanthos_wm[name].values.transpose())

            # xanthos
            self.ini_channel_storage = one_dim('Initial_Channel_Storage_Natural')
            self.sm = one_dim('Initial_SoilMoisture')
            self.instream_flow_natural = one_dim('Initial_instream_flow_Natural')
            # general reserviors
            self.purpose = one_dim('Main_Use')
            self.capacity = one_dim('Capacity')
            self.mtif_natural = one_dim('mtifl_natural')
            self.maxtif_natural = one_dim('Qmax_Turbine')
            # hydropower reservoir
            self.installed_cap = one_dim('ECAP')
            self.surface_area = one_dim('Surf_Area_SKM')
            self.max_depth = one_dim('Dam_HGT', np.nan, keep_nan=True)
            self.hp_release = Xanthos_wm['HP_Release'].values if 'HP_Release' in Xanthos_wm else None
            # water consumption and demand
            self.water_consumption = monthly('Total_Water_Consumption')
            self.total_demand_mmpermonth = monthly('Total_Water_Demand')
            self.total_demand_cumecs = self.total_demand_mmpermonth
            # grdc stations
            self.grdc_coord_index_file = pd.read_csv(
                                         self.config_obj.grdc_coord_index_file)
            # optimal params
            if self.config_obj.optimal_parameters_file is not None:
                self.optimal_parameters = self.load_data(
                                          self.config_obj.optimal_parameters_file)
