import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def run_ml_backtest():
    """
    Backtests the pairs trading strategy, using a machine learning model to filter trades.
    """
    DATA_DIR = 'kalman_with_features'
    OUTPUT_FILE = 'ml_backtest_results.csv' 
    INITIAL_CASH = 1_000_000.00
    COMMISSION = 0.001

    ENTRY_Z = 2.5
    EXIT_Z = 0.5
    MAX_POSITIONS = 20
    STOP_LOSS_DAYS = 100
    
    all_pairs = {}
    data_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        
    for file in data_files:
        pair = os.path.basename(file).replace('.csv', '')
        all_pairs[pair] = pd.read_csv(file, index_col='datetime', parse_dates=True)

    portfolio = {'cash': INITIAL_CASH, 'value': INITIAL_CASH, 'positions': {}}
    trade_log = []
    
    all_dates = sorted(list(set.union(*[set(df.index) for df in all_pairs.values()])))

    for date in tqdm(all_dates, desc="Backtesting"):
        current_value = portfolio['cash']
        open_pos_count = 0
        for pair, pos in portfolio['positions'].items():
            if pos:
                open_pos_count += 1
                try:
                    data = all_pairs[pair].loc[date]
                    current_value += (pos['s1_shares'] * data['stock1_price'] + pos['s2_shares'] * data['stock2_price'])
                except KeyError:
                    current_value += pos.get('entry_value', 0)
        
        portfolio['value'] = current_value
        trade_capital = portfolio['value'] / MAX_POSITIONS

        for pair, data in all_pairs.items():
            if date not in data.index:
                continue
                
            day_data = data.loc[date]
            pos = portfolio['positions'].get(pair)

            if pd.isna(day_data['z_score']) or pd.isna(day_data['hedge_ratio']):
                continue

            if pos:
                exit = False
                days_in_trade = (date.date() - pos['entry_date']).days
                
                if ((pos['type'] == 'Long' and day_data['z_score'] >= -EXIT_Z) or
                    (pos['type'] == 'Short' and day_data['z_score'] <= EXIT_Z)):
                    exit = True
                elif days_in_trade > STOP_LOSS_DAYS:
                    exit = True
                
                if exit:
                    s1_exit = pos['s1_shares'] * day_data['stock1_price']
                    s2_exit = pos['s2_shares'] * day_data['stock2_price']
                    exit_comm = (abs(s1_exit) + abs(s2_exit)) * COMMISSION
                    portfolio['cash'] += (s1_exit + s2_exit - exit_comm)
                    pnl = (s1_exit + s2_exit) - pos['entry_value']
                    pnl_comm = pnl - pos['entry_commission'] - exit_comm
                    
                    trade_log.append({'pair': pair, 'pnlcomm': pnl_comm, 'entry_date': pos['entry_date'], 'exit_date': date.date()})
                    
                    portfolio['positions'][pair] = None

            elif open_pos_count < MAX_POSITIONS:
                trade_type = None
                if day_data['z_score'] < -ENTRY_Z: trade_type = "Long"
                elif day_data['z_score'] > ENTRY_Z: trade_type = "Short"
                
                if trade_type:
                    s1_shares = (trade_capital * 0.5) / day_data['stock1_price']
                    if trade_type == 'Short': s1_shares *= -1
                    
                    s2_shares = -s1_shares * day_data['hedge_ratio']
                    s1_entry = s1_shares * day_data['stock1_price']
                    s2_entry = s2_shares * day_data['stock2_price']
                    entry_comm = (abs(s1_entry) + abs(s2_entry)) * COMMISSION
                    
                    portfolio['cash'] -= (s1_entry + s2_entry + entry_comm)
                    
                    portfolio['positions'][pair] = {
                        'type': trade_type, 'entry_date': date.date(),
                        's1_shares': s1_shares, 's2_shares': s2_shares,
                        'entry_value': s1_entry + s2_entry,
                        'entry_commission': entry_comm
                    }
                    open_pos_count += 1

    if trade_log:
        pd.DataFrame(trade_log).to_csv(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    run_ml_backtest()