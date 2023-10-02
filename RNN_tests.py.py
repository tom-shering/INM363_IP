# REF: https://gist.github.com/krishnagists/39b76e9777219ab3a88b

import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_utils import normalise_dataframe, denormalise_data
from data_utils import create_time_sequences_and_targets, create_dataloaders


class TestNormalisation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_df = pd.DataFrame({"Saturn": [2.8, 7.2, 1.1], "Mercury": [1852.6, 1853.46, 1852.128]})
        cls.columns_to_be_normalised = ["Saturn", "Mercury"]
        cls.normalised_df, cls.scalar_values = normalise_dataframe(cls.test_df, cls.columns_to_be_normalised)
    
    def test_returned_df_has_same_dimensions(self):
        self.assertEqual(self.normalised_df.shape, self.test_df.shape)
        self.assertListEqual(list(self.normalised_df.columns), list(self.test_df.columns))
    
    def test_returned_columns_are_floats(self):
        for column in self.columns_to_be_normalised:
            self.assertTrue(np.issubdtype(self.normalised_df[column].dtype, np.floating))

    # Floating point errors are accounted for below with np.isclose. 
    # REF: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    def test_normalisation_bounds(self):
        for column in self.columns_to_be_normalised:
            self.assertTrue(np.isclose(self.normalised_df[column].max(), 1, atol=1e-6) or
                            self.normalised_df[column].max() < 1)
            self.assertTrue(np.isclose(self.normalised_df[column].min(), 0, atol=1e-6) or
                            self.normalised_df[column].min() > 0)

            
class TestDenormalisation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_df = pd.DataFrame({"Saturn": [2.8, 7.2, 1.1], "Mercury": [1852.6, 1853.46, 1852.128]})
        cls.columns_to_be_normalised = ["Saturn", "Mercury"]
        cls.dummy_forecast_normalised_df = pd.DataFrame({"Saturn": [0.2, 1.0, 0.0], "Mercury": [0.34, 1.0, 0.00]})
        cls.normalised_df, cls.scalar_values = normalise_dataframe(cls.test_df, cls.columns_to_be_normalised)
        # tru = dummy true values, fc = forecast.
        cls.den_tru_sat, cls.den_fc_sat = denormalise_data(cls.normalised_df["Saturn"].values, 
                                                           cls.dummy_forecast_normalised_df["Saturn"].values, 
                                                           cls.scalar_values, "Saturn")
        cls.den_tru_mer, cls.den_fc_mer = denormalise_data(cls.normalised_df["Mercury"].values, 
                                                           cls.dummy_forecast_normalised_df["Mercury"].values, 
                                                           cls.scalar_values, "Mercury")
        cls.returned_series = [cls.den_tru_sat, cls.den_fc_sat, cls.den_tru_mer, cls.den_fc_mer]
    
    def test_denorm_shape_matches_norm(self):
        self.assertEqual(self.den_tru_sat.shape, self.normalised_df["Saturn"].values.shape)
        self.assertEqual(self.den_tru_mer.shape, self.normalised_df["Mercury"].values.shape)
    
    def test_returned_series_are_floats(self):
        for series in self.returned_series:  
            self.assertTrue(np.issubdtype(series.dtype, np.floating))
    
    def test_denormalised_values_are_same_as_original(self):
        self.assertTrue(np.all(np.isclose(self.test_df["Saturn"].values, self.den_tru_sat)))
        self.assertTrue(np.all(np.isclose(self.test_df["Mercury"].values, self.den_tru_mer)))
    
    def test_denormalised_forecast_values_are_scaled_back_up(self):
        self.assertTrue(np.all(self.den_fc_sat < 0) | np.all(self.den_fc_sat > 1))
        self.assertTrue(np.all(self.den_fc_mer < 0) | np.all(self.den_fc_mer > 1))
        
        

class TestTimeSeriesSequenceExtraction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls, data=None):
        if data is not None:
            cls.time_series_data = data 
        else:
            cls.time_series_data = np.random.rand(100, 6)  # Synthetic data can be used for stand-alone testing. 
        cls.sequence_length = 24
        cls.X, cls.y = create_time_sequences_and_targets(cls.time_series_data, cls.sequence_length)
    
    # @classmethod
    # def tearDownClass(cls):
    #   There is not actually anything that needs sweeping up after these tests, but including as a placeholder. 
    
    # The data needs to be a numpy array to be processed effectively:
    def test_array_type(self):
        self.assertIsInstance(self.X, np.ndarray)
        self.assertIsInstance(self.y, np.ndarray)
    
    # The data should have a shape that is equal to the following lengths to be processed as I expect.
    # Shape dimensions are dynamic to accommodate future changes. 
    def test_shapes(self):
        self.assertEqual(self.X.shape, (len(self.time_series_data) - self.sequence_length, self.sequence_length,
                                        self.time_series_data.shape[1]))
        self.assertEqual(self.y.shape, (len(self.time_series_data) - self.sequence_length,))
    
    # Testing that the target value is the next value after the sequence, in the first column, which corresponds
    # ...to the location of the variable to be forecast.  
    def test_target_extraction(self):
        for i in range(len(self.y)):
            self.assertEqual(self.y[i], self.time_series_data[i + self.sequence_length][0])



class TestDataloaderInitialisation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 50
        cls.X_train_synth = torch.randn(2000, 6) # j samples with k features. 
        cls.y_train_synth = torch.randn(2000, 1)
        cls.X_val_synth = torch.randn(1000, 6)
        cls.y_val_synth = torch.randn(1000, 1)
        cls.X_test_synth = torch.randn(1000, 6)
        cls.y_test_synth = torch.randn(1000, 1)
        cls.train_loader, cls.val_loader, cls.test_loader = create_dataloaders(cls.X_train_synth, 
                                                                               cls.y_train_synth, 
                                                                               cls.X_val_synth, 
                                                                               cls.y_val_synth, 
                                                                               cls.X_test_synth,
                                                                               cls.y_test_synth,
                                                                               cls.batch_size)
    # @classmethod
    # def tearDownClass(cls):
    #   There is not actually anything that needs sweeping up after these tests, but including as a placeholder. 
        
    def test_train_loader_length(self):
        expected_length = len(self.X_train_synth) // self.batch_size  
        self.assertEqual(len(self.train_loader), expected_length)
    
    def test_val_loader_length(self):
        expected_length = len(self.X_val_synth) // self.batch_size
        self.assertEqual(len(self.val_loader), expected_length)
        
    def test_test_loader_length(self):
        expected_length = len(self.X_test_synth) // self.batch_size
        self.assertEqual(len(self.test_loader), expected_length)
            
if __name__ == "__main__":
    unittest.main()

