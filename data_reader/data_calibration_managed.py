import xarray
import numpy as np
import scipy.io as scio
from xanthos.data_reader.data_reference import DataReference


class DataCalibrationManaged(DataReference):
    """Load data for calibration that uses streamflow and accounts for water management."""

    def __init__(self,
                 config_obj=None,
                 cal_observed=None,
                 scaler_observed = None,
                 abcdm_params = None,
                 purpose_file=None,
                 capacity_file=None,
                 hp_release_file=None,
                 water_consumption_file=None,
                 instream_flow_natural_file=None,
                 initial_channel_storage_natural_file=None,
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
            self.purpose = np.load(purpose_file)
            self.capacity = np.load(capacity_file)
            self.hp_release = np.load(hp_release_file)
            self.water_consumption = np.load(water_consumption_file)
            self.instream_flow_natural = np.load(instream_flow_natural_file)
            self.ini_channel_storage = np.load(initial_channel_storage_natural_file)
            self.sm = np.load(sm_file)
            self.mtif_natural = np.load(mtif_natural_file)
            self.maxtif_natural = np.load(maxtif_natural_file)
            self.total_demand_cumecs = np.load(total_demand_cumecs_file)
            self.grdc_coord_index_file = np.load(grdc_coord_index_file)


        else:

            self.config_obj = config_obj
            self.start_year = config_obj.StartYear
            self.end_year = config_obj.EndYear
            self.nmonths = config_obj.nmonths
            self.set_calibrate = config_obj.set_calibrate
            self.scaler_observed = self.load_data(self.config_obj.scaler_observed, 0)
            self.abcdm_params = self.load_data(self.config_obj.abcdm_params, 0) 
            
            super().__init__(nmonths=self.nmonths)

            # use basin-level flow as target for calibration; select only columns for basin number and runoff
            try:
                if self.set_calibrate == 0:
                    self.cal_obs = self.load_data(self.config_obj.cal_observed, 0)            
                else:
                    self.cal_obs = self.load_data(self.config_obj.cal_observed, 0)[:, [0, 3]]    
                                       
            except AttributeError:
                pass

            # load xanthos wm file
            Xanthos_wm =  xarray.open_dataset(self.config_obj.Xanthos_wm_file)
			# xanthos 
            self.ini_channel_storage = Xanthos_wm.Main_Use.values
            self.sm = Xanthos_wm.Initial_SoilMoisture.values	
            self.instream_flow_natural = Xanthos_wm.Main_Use.values			
			# general reserviors 
            self.purpose = Xanthos_wm.Main_Use.values
            self.capacity = Xanthos_wm.Capacity.values
            self.mtif_natural = Xanthos_wm.Main_Use.values
            self.maxtif_natural = Xanthos_wm.Qmax_Turbine.values
            # hydropower reservoir 
            self.installed_cap = Xanthos_wm.ECAP.values
            self.surface_area = Xanthos_wm.Surf_Area_SKM.values
            self.max_depth = Xanthos_wm.Dam_HGT.values
			# water consumption and demand
            self.water_consumption = Xanthos_wm.Total_Water_Consumption.values.transpose()			
            self.total_demand_mmpermonth = Xanthos_wm.Total_Water_Demand.values.transpose()
			# grdc stations 
            self.grdc_coord_index_file = scio.loadmat(self.config_obj.grdc_coord_index_file)['GRDC_xanthosCoordIndx']
