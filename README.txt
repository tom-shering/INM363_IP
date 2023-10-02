README


General Code Structure is given in section 3.12 of the report. 

energy_preprocessing and weather_preprocessing files import raw data (energy_dataset.csv and weather_features.csv) and export processed data: energy_mv.csv and weather_df_1.csv.
All other notebooks use total_df_mv.csv (mv=multivariate). This file is a combined, processed file. The preprocessing_and_SD.ipynb file performs seasonal decomposition and is equivalent to the 'seasonal' decomposition notebook in the report. The 'central' notebooks correspond to example versions of the central notebook mentioned in section 3.12. 

In order to run these files, key packages that need to be installed include torch, sns, matplotlib, statsmodels, unittest, contextlib, warnings, collections and IPython. 

The 'central' notebooks are example notebooks used from experimentation stage. All experiments reported in the project can be run from notebooks like this, 
but with simple changes to exogenous variable use, hyperparameter setup, etc. 

The MASE_evaluation file takes all provided energy data from the energy_MASE.csv file and calculates MASE values for ENTSO-E.

Requirements files for all notebooks are included. 

