import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
import joblib

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

def get_vix_data(filepath='data/vix_data.csv'):
    """
    Loads and preprocesses the VIX data.
    """
    try:
        vix_df = pd.read_csv(filepath, index_col='datetime', parse_dates=True).sort_index()
        vix_df = vix_df[['close']].rename(columns={'close': 'vix'})
        
        # Calculate a rolling Z-score for VIX to identify high/low volatility regimes
        vix_df['vix_zscore'] = (vix_df['vix'] - vix_df['vix'].rolling(60).mean()) / vix_df['vix'].rolling(60).std()
        
        return vix_df
    except FileNotFoundError:
        print(f"Warning: VIX data not found at '{filepath}'. VIX features will not be used.")
        return None

def train_model(trades_filepath='detailed_trades_features.csv', vix_data=None):
    """
    Trains a Random Forest model to predict trade profitability.
    """
    print("--- Starting Profitability Model Training ---")
    try:
        trades_df = pd.read_csv(trades_filepath)
    except FileNotFoundError:
        print(f"Error: Trade log '{trades_filepath}' not found. Please run the backtester first.")
        return

    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['is_profitable'] = (trades_df['pnlcomm'] > 0).astype(int)

    # Merge with VIX data if available
    if vix_data is not None:
        vix_data.index = vix_data.index.normalize()
        trades_df = pd.merge(trades_df, vix_data, left_on='entry_date', right_index=True, how='left')
    else:
        trades_df['vix_zscore'] = np.nan
    
    trades_df['high_vix'] = (trades_df['vix_zscore'] > 1).astype(int)
    trades_df['low_vix'] = (trades_df['vix_zscore'] < -1).astype(int)

    # Define features and target
    features = [
        'spread_volatility', 'hedge_ratio', 'correlation_30d',
        'stock1_rsi', 'stock1_atr_norm',
        'stock2_rsi', 'stock2_atr_norm',
        'high_vix', 'low_vix'
    ]
    target = 'is_profitable'

    # Ensure all feature columns exist, filling with NaN if necessary
    for col in features:
        if col not in trades_df.columns:
            trades_df[col] = np.nan

    # Split data into training and testing sets based on a fixed date
    train_df = trades_df[trades_df['entry_date'] <= '2022-12-31']
    test_df = trades_df[(trades_df['entry_date'] > '2022-12-31') & (trades_df['entry_date'] <= '2024-07-31')]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # Handle missing values using an imputer
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=features)
    
    # --- START OF THE FIX ---
    # Check the distribution of classes in the training data before applying SMOTE
    print("\nProfit/Loss distribution in the training data:")
    print(y_train.value_counts())

    # Only apply SMOTE if there is more than one class to balance
    if len(y_train.unique()) > 1:
        print("\nApplying SMOTE to balance the training data...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
        print("SMOTE applied successfully.")
    else:
        # If only one class is present, skip SMOTE and use the original data
        print("\nWarning: Only one class (all profits or all losses) found in the training data.")
        print("SMOTE will be skipped. The model will be trained on the imbalanced data.")
        X_train_resampled, y_train_resampled = X_train_imputed, y_train
    # --- END OF THE FIX ---
    
    # Train the Random Forest model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_resampled, y_train_resampled)
    print("Model training complete.")

    # Evaluate the model on the out-of-sample test data
    print("\n--- Model Evaluation on Test Set ---")
    y_pred = rf_model.predict(X_test_imputed)
    print(classification_report(y_test, y_pred, target_names=['Loss', 'Profit']))
    
    # Save the trained model and the imputer for later use in the forward test
    print("\nSaving model and imputer...")
    joblib.dump(rf_model, 'profitability_model.joblib')
    joblib.dump(imputer, 'imputer.joblib')
    print("✅ Model and imputer saved successfully.")

if __name__ == '__main__':
    vix_features = get_vix_data()
    train_model(vix_data=vix_features)