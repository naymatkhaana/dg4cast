
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Define function for SMAPE calculation
def smape(true_values, predicted_values):
    return 100 * np.mean(2 * np.abs(predicted_values - true_values) / (np.abs(predicted_values) + np.abs(true_values)))

# Define SARIMA forecasting function
def sarima_forecast(train_series, horizon, model):
    """
    Inputs:
        train_series (array-like): The training part of the time series data.
        horizon (int): The number of steps to forecast.
        model (SARIMAXResults): The fitted SARIMA model.
    Outputs:
        forecast (array-like): The forecasted values for the given horizon.
    """
    forecast = model.get_forecast(steps=horizon)
    # forecast = sarima_fit.get_prediction(start=len(train), end=len(train) + forecast_steps - 1, dynamic=True)
    return forecast.predicted_mean

# Define function to evaluate SARIMA model with rolling window
def evaluate_sarima(time_series, order, seasonal_order, horizon):
    """
    Inputs:
        time_series (array-like): The full input time series data.
        order (tuple): ARIMA (p, d, q) order.
        seasonal_order (tuple): Seasonal (P, D, Q, s) order.
        horizon (int): The individual forecast horizon (steps ahead).
    Outputs:
        Plots of estimated vs true values on the entire test set and prints MSE, MAE, and SMAPE.
    """
    # Split the time series into training (80%) and test sets
    split_idx = int(0.8 * len(time_series))
    train_series = time_series[:split_idx]
    test_series = time_series[split_idx:]

    predictions = np.zeros((len(test_series) - horizon,horizon))
    true_values = np.zeros((len(test_series) - horizon,horizon))

    for i in range(len(test_series) - horizon):
        # Fit SARIMA on rolling window
        # print(i)
        model = SARIMAX(time_series[:split_idx + i], order=order, seasonal_order=seasonal_order)
        model_fitted = model.fit(disp=False)
        forecast = sarima_forecast(time_series[:split_idx + i], horizon, model_fitted)
        predictions[i,:] = (forecast)  # Take the final step forecast
        true_values[i,:] = (test_series[i:i + horizon,0])

    # Convert to arrays for metric calculation
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    smape_list = []
    for h in range(horizon):
        trues = true_values[horizon:,h]
        preds = predictions[horizon:,h]

        # true_values = true_values[:,0]
        # print(true_values.shape)
        # print(predictions)
        # Compute metrics
        trues = trues.reshape((len(trues),1))
        preds = preds.reshape((len(preds),1))

        # print("true_values.shape:", true_values.shape)
        # print(np.zeros((len(true_values),2)).shape)
        trues = np.hstack((trues,np.zeros((len(trues),2)) ))
        preds = np.hstack((preds,np.zeros((len(preds),2)) ))

        trues = scaler.inverse_transform(trues)
        preds = scaler.inverse_transform(preds)
        trues = trues[:,0]
        preds = preds[:,0]

        trues = np.expm1(trues)
        preds = np.expm1(preds)

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        smape_value = smape(trues, preds)
        smape_list.append(smape_value)

        # Print evaluation metrics
        print(f"current horizon: {h}")        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"SMAPE: {smape_value:.4f}%")

        """
        # Plot true vs forecasted values for the test set
        plt.figure(figsize=(10, 6))
        plt.plot(range(split_idx + horizon, len(time_series) - horizon), trues, label='True Values', color='blue', marker='o', linestyle='-')
        plt.plot(range(split_idx + horizon, len(time_series) - horizon), preds, label='SARIMA Forecast', color='red', linestyle='--', marker='x')
        plt.title(f'SARIMA Forecast vs True Values (Test Set)')
        plt.xlabel('Time Steps (Test Set)')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        """

    return np.mean(np.array(smape_list)) 

# Example usage
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('-p', '--p', type=int, required=True, help="p")
    parser.add_argument('-q', '--q', type=int, required=True, help="q")
    parser.add_argument('-P', '--P', type=int, required=True, help="P")
    parser.add_argument('-Q', '--Q', type=int, required=True, help="Q")

    args = parser.parse_args()
    
    # Load your dataset
    df = pd.read_csv('national_illness_24.csv')

    time_series = np.log1p(df['ILITOTAL'].values).reshape(-1,1)  # Log-transform for stabilization
    scaler.fit(time_series)
    time_series = scaler.transform(time_series)

    # Define SARIMA parameters (p, d, q) and seasonal (P, D, Q, s)
    
    p=args.p
    d=0
    q=args.q
    P=args.P
    D=0
    Q=args.Q
    S=13
    
    for p in range(1,5):
        for q in range(1,5):
            for P in range(0,1):
                for Q in range(0,1):

                    print("p: ", p)
                    print("d: ", d)
                    print("q: ", q)
                    print("P: ", P)
                    print("D: ", D)
                    print("Q: ", Q)
                    print("S: ", S)


                    order = (p,d,q)
                    seasonal_order = (P,D,Q,S)  # seasonality (s=12)

                    # Evaluate SARIMA model for different horizons
                    for horizon in range(12, 13):
                        smape_value = evaluate_sarima(time_series, order, seasonal_order, horizon)
                        print(f"mean SMAPE Value: {smape_value:.4f} | Horizon: {horizon}")
