import pandas as pd
import numpy as np
import glob
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

def merge_price_data(path="/Users/shreyasshivapuji/Desktop/pairs-trading/data"):
    """
    Finds, loads, and merges all 'sp500_prices_*.csv' files.
    """
    csv_files = glob.glob(os.path.join(path, "sp500_prices_*.csv"))
    
    if not csv_files:
        print("Error: No 'sp500_prices_*.csv' files found.")
        return None

    df_list = [pd.read_csv(filename, index_col='datetime', parse_dates=True) for filename in sorted(csv_files)]
    
    combined_df = pd.concat(df_list, axis=1)
    
    combined_df.index = pd.to_datetime(combined_df.index).tz_localize(None).normalize()
    
    return combined_df

def discover_correlated_pairs(df, min_correlation=0.8):
    """
    Finds pairs of stocks with a correlation above a given threshold.
    """
    if df is None:
        return None
        
    corr_matrix = df.corr()
    
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    strong_pairs = upper_triangle[abs(upper_triangle) > min_correlation].stack().reset_index()
    strong_pairs.columns = ['Stock A', 'Stock B', 'Correlation']
    
    formatted_pairs = strong_pairs.apply(lambda row: f"{row['Stock A']}-{row['Stock B']}", axis=1)
    
    return pd.DataFrame(formatted_pairs, columns=['Correlated Pairs'])

def identify_cointegrated_pairs(price_data, correlated_pairs_file='highly_correlated_pairs.csv'):
    """
    Tests correlated pairs for cointegration.
    """
    try:
        pairs_df = pd.read_csv(correlated_pairs_file)
        pairs_to_test = pairs_df['Correlated Pairs'].tolist()
    except FileNotFoundError:
        print(f"Error: '{correlated_pairs_file}' not found.")
        return None

    cointegrated_pairs = []
    
    for pair in tqdm(pairs_to_test, desc="Testing Pairs for Cointegration"):
        try:
            stock1, stock2 = pair.split('-')
            
            stock1_prices = price_data[stock1]
            stock2_prices = price_data[stock2]
            
            combined_prices = pd.concat([stock1_prices, stock2_prices], axis=1).dropna()
            
            if len(combined_prices) > 100:
                _, p_value, _ = coint(combined_prices[stock1], combined_prices[stock2])
                
                if p_value < 0.05:
                    cointegrated_pairs.append({'Pair': pair, 'p-value': p_value})
        except (KeyError, Exception):
            continue
            
    return pd.DataFrame(cointegrated_pairs)

def calculate_half_life(spread):
    """
    Calculates the half-life of a mean-reverting time series using the
    Ornstein-Uhlenbeck formula.
    """
    # Lag the spread and calculate the difference
    spread_lag = spread.shift(1).dropna()
    spread_ret = (spread - spread_lag).dropna()
    
    # Use OLS to regress the spread returns on the lagged spread
    # This is equivalent to fitting the Ornstein-Uhlenbeck process
    model = sm.OLS(spread_ret, sm.add_constant(spread_lag))
    result = model.fit()
    
    # The half-life is -log(2) divided by the regression coefficient
    half_life = -np.log(2) / result.params.iloc[1]
    
    return half_life

def filter_by_half_life(price_data, cointegrated_pairs_df, min_half_life=5, max_half_life=100):
    """
    Calculates and filters pairs based on the half-life of their spread.
    The final DataFrame includes Pair, p-value, and Half-Life.
    """
    if cointegrated_pairs_df is None or cointegrated_pairs_df.empty:
        return None
    
    print(f"\nCalculating half-life for {len(cointegrated_pairs_df)} cointegrated pairs...")
    final_candidates = []

    for _, row in tqdm(cointegrated_pairs_df.iterrows(), total=len(cointegrated_pairs_df), desc="Calculating Half-Life"):
        pair = row['Pair']
        p_value = row['p-value']
        stock1, stock2 = pair.split('-')
        
        try:
            pair_prices = price_data[[stock1, stock2]].dropna()
            
            # Use OLS to find the hedge ratio and create the spread
            model = sm.OLS(pair_prices[stock1], sm.add_constant(pair_prices[stock2]))
            result = model.fit()
            hedge_ratio = result.params.iloc[1]
            spread = pair_prices[stock1] - hedge_ratio * pair_prices[stock2]
            
            hl = calculate_half_life(spread)
            
            # Filter pairs to be within the desired half-life range
            if min_half_life <= hl <= max_half_life:
                final_candidates.append({'Pair': pair, 'p-value': p_value, 'Half-Life': hl})
        except Exception:
            # Skip pairs that result in calculation errors
            continue

    print(f"Found {len(final_candidates)} pairs within the half-life range ({min_half_life}-{max_half_life} days).")
    return pd.DataFrame(final_candidates)

if __name__ == "__main__":
    # 1. Load and merge all price data
    all_prices = merge_price_data(path='./data') # Assuming data is in a 'data' subfolder
    
    # if all_prices is not None:
    #     # 2. Find highly correlated pairs
    #     correlated_pairs = discover_correlated_pairs(all_prices)
        
    #     if correlated_pairs is not None:
    #         correlated_pairs.to_csv('highly_correlated_pairs.csv', index=False)
            
    #         # 3. Test correlated pairs for cointegration
    #         cointegrated_pairs = identify_cointegrated_pairs(all_prices, 'highly_correlated_pairs.csv')
            
    #         if cointegrated_pairs is not None and not cointegrated_pairs.empty:
    #             cointegrated_pairs.to_csv('cointegrated_pairs.csv', index=False)

    cointegrated_pairs = pd.read_csv('cointegrated_pairs.csv')
    # 4. NEW STEP: Calculate half-life and filter the pairs
    final_candidates = filter_by_half_life(all_prices, cointegrated_pairs)
    
    if final_candidates is not None and not final_candidates.empty:
        # 5. Save the final list of candidate pairs
        final_candidates.sort_values('Half-Life', inplace=True)
        final_candidates.to_csv('final_candidate_pairs.csv', index=False)
        print("\nSuccessfully created 'final_candidate_pairs.csv'.")