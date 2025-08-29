import pandas as pd
import quantstats as qs
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys 

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore")

def load_and_process_tradelog(filename):
    """
    Loads the trade log CSV and prepares it for analysis.
    """
    if not os.path.exists(filename):
        print(f"❌ FATAL ERROR: The trade log '{filename}' was not found. Please run a backtest first.")
        return None

    df = pd.read_csv(filename)
    print(f"✅ Successfully loaded '{filename}' with {len(df)} trades.")
    
    # Clean and format the data
    df.dropna(subset=['entry_date', 'exit_date', 'pnlcomm'], inplace=True)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df['pnlcomm'] = pd.to_numeric(df['pnlcomm'])
    return df

def calculate_daily_returns(tradelog_df, initial_capital=1_000_000):
    """
    Converts a log of trades into a daily returns series for quantstats.
    """
    if tradelog_df is None or tradelog_df.empty:
        return pd.Series(dtype=float)

    start_date = tradelog_df['entry_date'].min()
    end_date = tradelog_df['exit_date'].max()
    
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.Series(dtype=float)

    # Create a full business-day date range for the backtest period
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    daily_pnl = pd.Series(0.0, index=date_range)
    
    # Sum up P&L for each day trades were closed
    pnl_by_exit_date = tradelog_df.groupby('exit_date')['pnlcomm'].sum()
    daily_pnl.update(pnl_by_exit_date)
    
    # Calculate the equity curve and daily percentage returns
    equity_curve = initial_capital + daily_pnl.cumsum()
    daily_returns = equity_curve.pct_change().fillna(0)
    
    return daily_returns

# MODIFICATION: Added 'output_filename' parameter to the function definition
def display_performance_results(strategy_name, daily_returns, tradelog_df, risk_free_rate=0.0, output_filename=None):
    """
    Calculates and prints detailed performance metrics and displays plots.
    Also saves the text report and plot to files if a filename is provided.
    """
    # --- MODIFICATION: Logic to write text output to a file ---
    if output_filename:
        # Ensure we have a .txt extension for the report
        report_file_path = os.path.splitext(output_filename)[0] + '.txt'
        original_stdout = sys.stdout  # Save a reference to the original standard output
        print(f"\n📝 Writing performance report to '{report_file_path}'...")
        f = open(report_file_path, 'w')
        sys.stdout = f # Redirect stdout to the file
    
    # --- The original display logic starts here ---
    # All 'print' statements from here will go to the file if redirected
    try:
        print("\n" + "="*60)
        print(f"PERFORMANCE ANALYSIS: {strategy_name}")
        print("="*60)

        # Display Key Metrics using QuantStats
        qs.reports.metrics(daily_returns, mode='full', rf=risk_free_rate)

        # Display Trade-Specific Statistics
        print("\n--- Trade Statistics ---")
        total_trades = len(tradelog_df)
        winning_trades = tradelog_df[tradelog_df['pnlcomm'] > 0]
        losing_trades = tradelog_df[tradelog_df['pnlcomm'] <= 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        avg_win = winning_trades['pnlcomm'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnlcomm'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnlcomm'].sum() / losing_trades['pnlcomm'].sum()) if len(losing_trades) > 0 and losing_trades['pnlcomm'].sum() != 0 else np.inf

        print(f"Total Trades:       {total_trades}")
        print(f"Win Rate:           {win_rate:.2f}%")
        print(f"Average Win:        ${avg_win:,.2f}")
        print(f"Average Loss:       ${avg_loss:,.2f}")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print("="*60)
    
    finally:
        # --- MODIFICATION: Restore original stdout ---
        if output_filename:
            sys.stdout = original_stdout # Restore stdout
            f.close()
            # This print statement will now go to the console
            print("✅ Report saved successfully.")

    # --- Generate, Show, and Save Helpful Plots ---
    print("\n📈 Generating performance plots...")
    
    # Use quantstats to create a single figure with multiple plots
    fig = qs.plots.snapshot(daily_returns, title=f"{strategy_name} - Performance Snapshot", show=False, rf=risk_free_rate)
    
    # --- MODIFICATION: Save the plot to a file ---
    if output_filename:
        plot_file_path = os.path.splitext(output_filename)[0] + '.png'
        print(f"🖼️ Saving performance plot to '{plot_file_path}'...")
        fig.savefig(plot_file_path, dpi=300, bbox_inches='tight')
        print("✅ Plot saved successfully.")

    plt.show() # This command displays all generated plots

def main():
    """
    Main execution function to run the performance analysis.
    """
    # --- Configuration ---
    STRATEGY_NAME = 'ML-Enhanced Kalman Filter Strategy'
    TRADE_LOG_FILE = 'ml_backtest_results.csv'
    INITIAL_CAPITAL = 1_000_000
    RISK_FREE_RATE = 0.01
    # MODIFICATION: Define a base name for the output report and plot
    OUTPUT_FILENAME = 'ml_strategy_report'


    # --- Analysis Pipeline ---
    tradelog_df = load_and_process_tradelog(TRADE_LOG_FILE)
    if tradelog_df is None:
        return # Stop execution if the trade log can't be loaded
        
    daily_returns = calculate_daily_returns(tradelog_df, INITIAL_CAPITAL)
    
    if daily_returns.empty:
        print("\n❌ Could not calculate daily returns. The trade log might be empty or contain invalid dates.")
        return
    
    # MODIFICATION: Pass the OUTPUT_FILENAME to the display function
    display_performance_results(STRATEGY_NAME, daily_returns, tradelog_df, RISK_FREE_RATE, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()