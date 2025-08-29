import pandas as pd
import numpy as np
import glob
import os
from pykalman import KalmanFilter
from tqdm import tqdm
import talib
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore")

# Define the date ranges for data processing
CALCULATION_START_DATE = '2019-09-01'
SIGNAL_START_DATE = '2020-01-01'
SIGNAL_END_DATE = '2025-07-31'

# --- Data Loading ---

def fetch_and_prepare_prices(path="/Users/shreyasshivapuji/Desktop/pairs-trading/data"):
    """
    Loads and merges all S&P 500 price data files from a specified directory.
    """
    files = glob.glob(os.path.join(path, "sp500_prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No price files found in '{path}'. Please check the directory.")
    
    dfs = [pd.read_csv(f, index_col='datetime', parse_dates=True) for f in sorted(files)]
    
    prices = pd.concat(dfs, axis=1)
    # Standardize index and remove duplicate columns/rows
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.loc[~prices.index.duplicated(keep='first')]
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices

# --- Signal and Feature Generation ---

def compute_kalman_signals(prices, stock1, stock2):
    """
    Generates core trading signals (z-score, hedge ratio) for a stock pair using a Kalman filter.
    """
    # Slice the data for the calculation period and remove any rows with missing values
    pair_data = prices.loc[CALCULATION_START_DATE:SIGNAL_END_DATE, [stock1, stock2]].dropna()
    if len(pair_data) < 100: # Ensure there is enough data for a stable calculation
        return None
    
    # Set up the observation matrix for the Kalman filter
    observation_matrix = np.vstack([
        pair_data[stock2].values, 
        np.ones(len(pair_data))
    ]).T[:, np.newaxis, :]

    # Initialize the Kalman Filter with optimized parameters
    kf = KalmanFilter(
        n_dim_state=2,
        n_dim_obs=1,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrix,
        observation_covariance=0.1,
        transition_covariance=1e-11 * np.eye(2)
    )
    
    # Apply the filter to get the state means (hedge ratio and intercept)
    state_means, _ = kf.filter(pair_data[stock1].values)
    
    # Calculate the dynamic spread and its rolling Z-score
    spread = pair_data[stock1] - (state_means[:, 0] * pair_data[stock2])
    z_score = (spread - spread.rolling(window=60).mean()) / spread.rolling(window=60).std()
    
    # Assemble the final signals DataFrame
    signals = pd.DataFrame({
        'stock1_price': pair_data[stock1],
        'stock2_price': pair_data[stock2],
        'z_score': z_score,
        'hedge_ratio': state_means[:, 0]
    }, index=pair_data.index)
    
    # Return only the data relevant for the backtest period
    return signals.loc[SIGNAL_START_DATE:SIGNAL_END_DATE]

def enrich_with_features(signals, prices, stock1, stock2):
    """
    Adds technical analysis and statistical features to the signal data for use in the ML model.
    """
    if signals is None or signals.empty:
        return None
    
    features = signals.copy()
    spread = features['stock1_price'] - (features['hedge_ratio'] * features['stock2_price'])
    
    # --- Feature Engineering ---
    # Technical indicators for each stock
    features['stock1_rsi'] = talib.RSI(features['stock1_price'], timeperiod=14)
    features['stock1_atr_norm'] = talib.ATR(features['stock1_price'], features['stock1_price'], features['stock1_price'], 14) / features['stock1_price']
    
    features['stock2_rsi'] = talib.RSI(features['stock2_price'], timeperiod=14)
    features['stock2_atr_norm'] = talib.ATR(features['stock2_price'], features['stock2_price'], features['stock2_price'], 14) / features['stock2_price']
    
    # Spread-based features
    features['spread_volatility'] = spread.diff().abs().rolling(window=30).std().shift(1)
    
    # Rolling correlation of the pair
    full_prices = prices.loc[CALCULATION_START_DATE:SIGNAL_END_DATE, [stock1, stock2]].dropna()
    features['correlation_30d'] = full_prices[stock1].rolling(window=30).corr(full_prices[stock2]).shift(1)
    
    return features

# --- Main Execution ---

def main(price_path="/Users/shreyasshivapuji/Desktop/pairs-trading/data", pairs_file='/Users/shreyasshivapuji/Desktop/pairs-trading/final_candidate_pairs.csv'):
    """
    Main pipeline to load data, generate signals, add features, and save the results.
    """
    try:
        prices = fetch_and_prepare_prices(price_path)
        pairs_df = pd.read_csv(pairs_file)
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        return
    
    # Create the directory for the output files
    output_dir = "kalman_with_features"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(pairs_df)} pairs from '{pairs_file}'...")
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Generating Signals & Features"):
        # --- THIS IS THE KEY CHANGE ---
        # Read from the 'Pair' column, which is in final_candidate_pairs.csv
        pair_string = row['Pair']
        stock1, stock2 = pair_string.split('-')
        
        try:
            # 1. Calculate the core signals using the Kalman Filter
            kalman_signals = compute_kalman_signals(prices, stock1, stock2)
            if kalman_signals is None:
                continue
                
            # 2. Enrich the signals with additional features for the ML model
            ml_features = enrich_with_features(kalman_signals.copy(), prices, stock1, stock2)
            if ml_features is not None:
                # Save the combined data to a CSV file for the backtester
                ml_features.to_csv(os.path.join(output_dir, f"{pair_string}.csv"))
                
        except Exception as e:
            print(f"Skipping pair {pair_string} due to an error: {str(e)}")
            continue

    print(f"\n✅ Signal and feature generation complete. Files saved in '{output_dir}'.")

if __name__ == "__main__":
    main()