import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def generate_training_tradelog():
    """
    Backtests the Kalman filter strategy on historical data to generate a detailed
    trade log with features. This log is then used to train the profitability prediction model.
    """
    # --- 1. CONFIGURATION ---
    DATA_DIR = 'kalman_with_features'
    OUTPUT_FILENAME = 'detailed_trades_features.csv' 
    INITIAL_CAPITAL = 1_000_000.00
    COMMISSION_RATE = 0.001
    
    # Strategy Parameters
    ENTRY_Z_SCORE = 2.5
    EXIT_Z_SCORE = 0.5
    MAX_CONCURRENT_PAIRS = 20
    MAX_HOLD_DAYS = 100
    
    # Feature columns to be captured at the time of entry
    FEATURE_COLUMNS = [
        'z_score', 'hedge_ratio', 'spread_volatility', 'correlation_30d',
        'stock1_rsi', 'stock1_atr_norm',
        'stock2_rsi', 'stock2_atr_norm'
    ]
    
    # --- 2. DATA LOADING ---
    all_pairs_data = {}
    data_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not data_files:
        print(f"Error: No data files found in '{DATA_DIR}'. Please run the signal_generator.py script first.")
        return
        
    print(f"Loading {len(data_files)} pair data files for backtest...")
    for file_path in data_files:
        pair_name = os.path.basename(file_path).replace('.csv', '')
        try:
            df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            all_pairs_data[pair_name] = df
        except Exception as e:
            print(f"Error loading {pair_name}: {str(e)}")
            continue

    # --- 3. PORTFOLIO AND TIMELINE SETUP ---
    portfolio = {
        'cash': INITIAL_CAPITAL,
        'total_value': INITIAL_CAPITAL,
        'positions': {} # Using a simpler structure for active positions
    }
    closed_trades = []
    
    # Create a master timeline of all unique trading days
    master_timeline = sorted(list(set.union(*[set(df.index) for df in all_pairs_data.values()])))
    print(f"\nBacktest will run from {master_timeline[0].date()} to {master_timeline[-1].date()}")

    # --- 4. MAIN BACKTEST LOOP ---
    for current_date in tqdm(master_timeline, desc="Backtesting to Generate Trades"):
        # A. Mark-to-Market portfolio and count open positions
        current_portfolio_value = portfolio['cash']
        open_positions_count = 0
        
        # We need a list of pairs to iterate over to avoid issues with modifying the dict while looping
        active_pairs = list(portfolio['positions'].keys())
        
        for pair in active_pairs:
            position = portfolio['positions'][pair]
            if position:
                open_positions_count += 1
                try:
                    # Update position value with current day's prices
                    data = all_pairs_data[pair].loc[current_date]
                    current_portfolio_value += (position['size_s1'] * data['stock1_price'] + 
                                                position['size_s2'] * data['stock2_price'])
                except KeyError:
                    # If today's price is not available, use the last known entry value
                    current_portfolio_value += position.get('entry_value', 0)
        
        portfolio['total_value'] = current_portfolio_value
        capital_per_trade = portfolio['total_value'] / MAX_CONCURRENT_PAIRS

        # B. Process signals for each pair
        for pair_name, pair_data in all_pairs_data.items():
            if current_date not in pair_data.index:
                continue
                
            daily_data = pair_data.loc[current_date]
            position = portfolio['positions'].get(pair_name)

            # Skip if essential data is missing
            if pd.isna(daily_data['z_score']) or pd.isna(daily_data['hedge_ratio']):
                continue

            # --- C. EXIT LOGIC ---
            if position:
                exit_signal, exit_reason = False, ""
                days_held = (current_date.date() - position['entry_date']).days
                
                # Exit conditions: Z-score reversion or time-based stop-loss
                if ((position['type'] == 'LONG' and daily_data['z_score'] >= -EXIT_Z_SCORE) or
                    (position['type'] == 'SHORT' and daily_data['z_score'] <= EXIT_Z_SCORE)):
                    exit_signal, exit_reason = True, "Z-Score Reversion"
                elif days_held > MAX_HOLD_DAYS:
                    exit_signal, exit_reason = True, "Time Stop Loss"
                
                if exit_signal:
                    exit_s1 = position['size_s1'] * daily_data['stock1_price']
                    exit_s2 = position['size_s2'] * daily_data['stock2_price']
                    exit_commission = (abs(exit_s1) + abs(exit_s2)) * COMMISSION_RATE
                    portfolio['cash'] += (exit_s1 + exit_s2 - exit_commission)
                    
                    pnl_before_commission = (exit_s1 + exit_s2) - position['entry_value']
                    pnl_after_commission = pnl_before_commission - position['entry_commission'] - exit_commission
                    
                    # Create a dictionary for the closed trade, including all features
                    trade_details = {
                        'pair': pair_name,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date.date(),
                        'days_held': days_held,
                        'exit_reason': exit_reason,
                        'pnl': pnl_before_commission,
                        'pnlcomm': pnl_after_commission,
                    }
                    # Add the features captured at entry
                    entry_features = {col: position['entry_features'].get(col, np.nan) for col in FEATURE_COLUMNS}
                    trade_details.update(entry_features)
                    closed_trades.append(trade_details)
                    
                    # Remove position from portfolio
                    del portfolio['positions'][pair_name]

            # --- D. ENTRY LOGIC ---
            elif open_positions_count < MAX_CONCURRENT_PAIRS:
                trade_type = None
                if daily_data['z_score'] < -ENTRY_Z_SCORE: trade_type = "LONG"
                elif daily_data['z_score'] > ENTRY_Z_SCORE: trade_type = "SHORT"
                
                if trade_type:
                    # Calculate position sizes based on allocated capital
                    size_s1 = (capital_per_trade * 0.5) / daily_data['stock1_price']
                    if trade_type == 'SHORT': size_s1 *= -1
                    
                    size_s2 = -size_s1 * daily_data['hedge_ratio']
                    
                    entry_s1_value = size_s1 * daily_data['stock1_price']
                    entry_s2_value = size_s2 * daily_data['stock2_price']
                    entry_commission = (abs(entry_s1_value) + abs(entry_s2_value)) * COMMISSION_RATE
                    
                    # Update cash and add position to portfolio
                    portfolio['cash'] -= (entry_s1_value + entry_s2_value + entry_commission)
                    
                    portfolio['positions'][pair_name] = {
                        'type': trade_type,
                        'entry_date': current_date.date(),
                        'size_s1': size_s1,
                        'size_s2': size_s2,
                        'entry_value': entry_s1_value + entry_s2_value,
                        'entry_commission': entry_commission,
                        'entry_features': {col: daily_data.get(col, np.nan) for col in FEATURE_COLUMNS}
                    }
                    open_positions_count += 1

    # --- 5. FINAL CLEANUP & SAVE ---
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        trades_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n✅ Backtest complete. Detailed trade log with features saved to '{OUTPUT_FILENAME}'.")
    else:
        print("\nNo trades were executed during the backtest period.")

if __name__ == '__main__':
    generate_training_tradelog()