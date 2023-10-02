import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_utils import generate_random_indices

def plot_losses(train_losses, val_losses, data_type, graph_title_note):
    
    plt.figure(figsize=(20, 12))
    sns.set_style("whitegrid")

    plt.plot(train_losses, label="Training Loss", color="#6495ED")
    plt.plot(val_losses, label="Validation Loss", color="#DAA520")

    plt.title(f"Training and Validation Losses over Epochs, for {data_type}, {graph_title_note}", fontsize=16)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

    
def plot_forecast_vs_true_values(true_values, forecast_values, data_type, graph_title_note, rolling_window=24, x_length=1000):
    
    #REF: https://pandas.pydata.org/docs/reference/api/pandas.Series.html
    # These need to be pd series so that .rolling() can be used.
    true_value_series = pd.Series(true_values)
    forecast_value_series = pd.Series(forecast_values)

    rolling_average_true = true_value_series.rolling(window=rolling_window).mean()
    rolling_average_forecast = forecast_value_series.rolling(window=rolling_window).mean()
    
    number_of_plots = int(np.ceil(len(rolling_average_forecast) / x_length))
    
    for n in range(number_of_plots):
        
        start_index = n * x_length
        end_index = (n + 1) * x_length
        
        plt.figure(figsize=(20, 12))
        sns.set_style("whitegrid")

        plt.plot(rolling_average_true[start_index:end_index], label="True Values", color="#008080")
        plt.plot(rolling_average_forecast[start_index:end_index], label="Forecast Values", color="#FF7F50")

        plt.title(f"True vs Forecast for {data_type} values, averaged with {rolling_window} Window, {graph_title_note}", fontsize=16)
        plt.xlabel("Hours")
        plt.ylabel(f"{data_type}")
        plt.legend()

        plt.show()
        
        
def plot_random_hourly_forecast_periods(true_values, forecast_values, data_type, graph_title_note):
    
    true_value_series = pd.Series(true_values)
    forecast_value_series = pd.Series(forecast_values)
    
    #random_indices = generate_random_indices(true_value_series, 4)
    # I've hard coded these so that different experiments can be compared directly. 
    random_indices = [2968, 3175, 4677, 6559] #Generated using 'random' package. 
    
    
    for start in random_indices:
        end = start + 24
        
        plt.figure(figsize=(20, 12))
        sns.set_style("whitegrid")

        plt.plot(true_value_series[start:end], label="True Values", color="#008080")
        plt.plot(forecast_value_series[start:end], label="Forecast Values", color="#FF7F50")

        plt.title(f"True vs Forecast for {data_type} values, {graph_title_note}", fontsize=16)
        plt.xlabel("Hours")
        plt.ylabel(f"{data_type}")
        plt.legend()

        plt.show()
        
def plot_errors(true_values, forecast_values, data_type, graph_title_note, is_it_test_data=True):
    
    if len(true_values) == len(forecast_values):
        errors = [pred - true for pred, true in zip(forecast_values, true_values)]
    else: 
        print("\n\n **WARNING** True and forecast values do not match in length. \n\n")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(20,12))
    plt.plot(errors, color="#CD5C5C")
    if is_it_test_data is not None:
        plt.title(f"Errors vs hour for test values for {data_type}, {graph_title_note}")
    else:
        plt.title(f"Errors vs hour for {data_type}, {graph_title_note}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Error")
    plt.show()
    

def style_dataframe(df):
    styles = [
        # Overall table:
        dict(selector="table", props=[("border", "2px solid #A9A9A9"),
                                      ("border-collapse", "collapse")]),
        
        # Headers:
        dict(selector="th", props=[("font-size", "12px"), 
                                   ("text-align", "center"),
                                   ("font-weight", "bold"),
                                   ("background-color", "#FAF0E6"),
                                   ("border", "1px solid #A9A9A9")]),
        
        # Cells:
        dict(selector="td", props=[("border", "1px solid #A9A9A9")]),
        
        # Row names: 
        dict(selector="th.index_name", props=[("font-size", "12px"),
                                              ("background-color", "#FAF0E6"),
                                              ("border", "1px solid #A9A9A9")])
    ]
    
    return df.style.set_table_styles(styles)


def display_table(data, title):
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data).T # .T transposes variables as rows.
    #df = df.applymap(lambda x: '{:.2e}'.format(x) if isinstance(x, (int, float)) else x)
    print(f"{title}")
    if "exp_num" in df.columns:
        df = df.set_index("exp_num")
    styled_df = style_dataframe(df)
    display(styled_df)
    print("\n")
    

def plot_decomposition(result):
    
    fig, axes = plt.subplots(4, 1, figsize=(12,8))
    
    result.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel("Observed")
    
    result.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel("Trend")
    
    result.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel("Seasonal")
    
    result.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel("Residual")
    
    plt.tight_layout()
    plt.show()