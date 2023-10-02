import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import DataLoader, TensorDataset

def normalise_dataframe(dataframe, columns_to_be_normalised):
    normalised_df = pd.DataFrame()
    scaler_values = {}

    for column in columns_to_be_normalised:
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalised_df[column] = scaler.fit_transform(dataframe[column].values.reshape(-1, 1)).flatten()
        scaler_values[column] = scaler
    
    return normalised_df, scaler_values


def denormalise_data(true_values, forecast_values, scaler_values, data_type):
# REF: https://stackoverflow.com/questions/49330195/how-to-use-inverse-transform-in-minmaxscaler-for-a-column-in-a-matrix
    scaler = scaler_values[data_type]

    denormalised_true = scaler.inverse_transform(np.array(true_values).reshape(-1, 1)).flatten()
    denormalised_forecast = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten()

    return denormalised_true, denormalised_forecast

def denormalise_data_sd(true_values, forecast_values, scaler_min, scaler_max):
   
    denormalised_true = true_values * (scaler_max - scaler_min) + scaler_min
    denormalised_forecast = forecast_values * (scaler_max - scaler_min) + scaler_min
    
    return denormalised_true, denormalised_forecast



def create_time_sequences_and_targets(time_series_data, sequence_length):
    sequences_list = []
    targets_list = []
    
    # This is taking data from all columns for each lot of sequence_length rows:
    for i in range(len(time_series_data) - sequence_length):
        sequence = time_series_data[i:i+sequence_length] # Taking data in sequence_length sized sections
        target_value = time_series_data[i+sequence_length][0] # The next value will be the target label
        sequences_list.append(sequence)
        targets_list.append(target_value)

    return np.array(sequences_list), np.array(targets_list)

def create_time_sequences_and_targets_mv(time_series_data, target_column_indices, sequence_length):
    sequences_list = []
    targets_list = []
    
    for i in range(len(time_series_data) - sequence_length):
        sequence = time_series_data[i:i+sequence_length]
        target_values = np.array([time_series_data[i+sequence_length, idx] for idx in target_column_indices])
        sequences_list.append(sequence)
        targets_list.append(target_values)

    return np.array(sequences_list), np.array(targets_list)

def create_time_sequences_and_targets_sd(time_series_data, target_column, exogenous_columns, sequence_length):
    
    sequences_list = []
    targets_list = []

    # Creating a list of column indices to be included in each sequence:
    sequence_columns = [target_column] + exogenous_columns
    sequence_column_indices = [time_series_data.columns.get_loc(col) for col in sequence_columns]
    
    # Getting the index for the target column:
    target_column_index = time_series_data.columns.get_loc(target_column)

    for i in range(len(time_series_data) - sequence_length):
        sequence = time_series_data.iloc[i:i+sequence_length, sequence_column_indices].values
        target_value = time_series_data.iloc[i+sequence_length, target_column_index]
        
        sequences_list.append(sequence)
        targets_list.append(target_value)

    return np.array(sequences_list), np.array(targets_list)




# At present this is only being used for testing but could be imported into main notebook. 
def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    
    # Converting data to PyTorch tensors:
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Creating TensorDatasets:
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    val_data = TensorDataset(X_val_tensor, y_val_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    # Creating DataLoaders:
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    # (I don't want to shuffle sequential data as the order matters)
    
    return train_loader, val_loader, test_loader


# Generates n non-overlapping indices for visualising:
def generate_random_indices(dataframe, number_of_indices_required):
    
    random_indices_list = []
    excluded_numbers = set()
    remaining_indices_available = range(0, len(dataframe) - 24)
    
    for indices in range(number_of_indices_required):
        random_index = np.random.choice(list(remaining_indices_available))
        random_indices_list.append(random_index)
        excluded_numbers.update(range(random_index, random_index + 24))
        remaining_indices_available = [i for i in range(0, len(dataframe) - 24) if i not in excluded_numbers]
        
    return random_indices_list

#IQR-normalised MAE:
def calculate_iqr_normalised_mae(true_values, forecast_values):
    mae = mean_absolute_error(true_values, forecast_values)
    iqr = np.percentile(true_values, 75) - np.percentile(true_values, 25)
    return (mae / iqr) * 100

# Quantifying validation loss trend towards the latter stages of training (validation loss trend):
def calculate_val_loss_trend(validation_losses, start_percentage=75):
    # start_percentage is the percentage of the way through training to begin trend analysis.
    start_index = int(len(validation_losses) * (start_percentage / 100))
    end_index = len(validation_losses) - 1  # adjusted to be last index of list 
    
    if start_index >= len(validation_losses) or start_index < 0 or start_index <= len(validation_losses) * 0.1:
        raise ValueError("The start_percentage should be a percentage towards the latter stages of training.")
    
    val_loss_latter_gradient = (validation_losses[end_index] - validation_losses[start_index]) / (end_index - start_index)
    
    return val_loss_latter_gradient


# Quantifying validation loss standard deviation (validation loss stability):
def calculate_val_loss_std(validation_losses, second_differential_threshold=0.1):
    # The second_differential_threshold is the quotient of the approximated second differential over the 
    # ... range, as a threshold to determine when the graph has stabilised near a stable value. 
    
    if len(validation_losses) <= 10:
        return "Insufficient Validation Loss Set Length"
    
    start_index = None
    val_loss_range = np.max(validation_losses) - np.min(validation_losses)
    threshold = val_loss_range * second_differential_threshold

    for index in range(10, len(validation_losses) - 1):
        local_gradient = validation_losses[index] - validation_losses[index - 1]
        preceding_local_gradient = validation_losses[index-1] - validation_losses[index - 2]
        approximated_second_differential = abs(preceding_local_gradient - local_gradient) 
        
        if approximated_second_differential <= threshold:
            start_index = index
            break 
        else:
            continue

    if start_index is not None:  
        values_to_be_evaluated = validation_losses[start_index:]
        validation_loss_std = np.std(values_to_be_evaluated)
        return validation_loss_std
    else:
        return "Validation losses do not reach a stable point as defined by threshold."

# Prediction bias function; essentially quantifying if the model over- or under-predicts in an unbalanced way:
def calculate_forecast_bias(predicted_values, true_values):
    predicted_values = np.array(predicted_values)
    true_values = np.array(true_values)
    return (predicted_values - true_values).mean()

# MASE
def calculate_mase(true_values, model_forecast_values):
    
    naive_forecast_values = np.roll(true_values, shift=1)
    
    naive_mae = mean_absolute_error(true_values[1:], naive_forecast_values[1:])
    model_mae = mean_absolute_error(true_values, model_forecast_values)
    
    mase = model_mae / naive_mae
    
    return mase

def take_logs_of_metrics_whilst_preserving_sign(value):
    if value > 0:
        return np.log(value + 1e-9)
    elif value < 0:
        return - np.log(abs(value) + 1e-9)
    else:
        return np.log(1e-9)




        