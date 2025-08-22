# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational finance project (YAP471 - Spring 2023-2024, TOBB University) focused on **Risk Analysis and Portfolio Creation with Genetic Algorithms using Statistical Analysis**. The project combines stock price prediction using machine learning models with genetic algorithm-based portfolio optimization.

## Key Dependencies

The project requires these Python packages (install via pip):
```bash
pip install pandas numpy yfinance tensorflow xgboost scikit-learn matplotlib seaborn ta tqdm
```

Core dependencies:
- **yfinance**: Stock data downloading
- **tensorflow/keras**: Deep learning models (CNN-LSTM, RNN)
- **xgboost**: Gradient boosting for stock prediction
- **ta**: Technical analysis indicators (MACD, RSI, Bollinger Bands, Ichimoku)
- **scikit-learn**: Machine learning utilities and preprocessing
- **tqdm**: Progress bars for genetic algorithm evolution

## Project Architecture

### 1. Stock Price Prediction Models
Located in `Stock-Price-Prediction/`:

- **CNN_LSTM_Stock_Price_Prediction.ipynb**: Hybrid CNN-LSTM model for time series prediction
- **RNN_Stock_Price_Prediction.ipynb**: Recurrent Neural Network approach
- **xgboost-stock-price-prediction.ipynb**: XGBoost regression model

### 2. Portfolio Optimization
- **finance_genetic.ipynb**: Genetic algorithm for portfolio weight optimization
- **Stock-Price-Prediction-Optimal-Portfolio.ipynb**: Main notebook integrating all three prediction models

### 3. Data Analysis
- **Explanatory-Data-Analysis/**: Contains exploratory analysis notebooks
- **Prediction_Datasets/**: CSV files with prediction results for major stocks (AAPL, GOOGL, TSLA, etc.)

## Common Development Workflow

### Running Stock Price Predictions

1. **Single Stock Analysis**: Use individual model notebooks in `Stock-Price-Prediction/`
   - Set `STOCK_NAME` variable (e.g., "AAPL", "GOOGL", "TSLA")
   - Run cells sequentially for data download, preprocessing, model training, and evaluation

2. **Multi-Model Comparison**: Use `Stock-Price-Prediction-Optimal-Portfolio.ipynb`
   - Automatically runs CNN-LSTM, RNN, and XGBoost on the same stock
   - Generates prediction datasets for portfolio optimization

### Portfolio Optimization with Genetic Algorithm

The genetic algorithm parameters are configurable:
```python
population_size = 1000
num_generations = 90
mutation_rate = 0.10-0.20
```

### Data Processing Pipeline

1. **Stock Data Download**: yfinance pulls historical data (2010-2024)
2. **Technical Indicators**: Automatically calculates moving averages, MACD, RSI, Bollinger Bands, Ichimoku
3. **Feature Engineering**: Creates windowed sequences for time series models
4. **Model Training**: Split data at 2020 for train/test (pre-COVID vs COVID era)
5. **Prediction Generation**: Models output future price predictions with volatility calculations

## Key Functions and Utilities

### Data Download and Processing
- `download_stock(stock_name)`: Downloads historical stock data
- `create_moving_average()`, `create_bollinger_bands()`, `create_macd()`, `create_rsi()`, `create_ichi()`: Technical indicators
- `window_data()`: Creates sliding windows for XGBoost features

### Model Utilities
- `create_pred_real_df()`: Combines predictions with actual prices and volatility
- `calculate_volatility()`: Computes rolling volatility from price series
- `val_train_loss_plot()`, `val_train_mse_plot()`: Training visualization

### Genetic Algorithm
- `fitness()`: Calculates Sharpe ratio for portfolio optimization
- Evolution loop with crossover, mutation, and elitism strategies

## Model Architectures

### CNN-LSTM Model
- TimeDistributed CNN layers (64→128→64 filters)
- Bidirectional LSTM layers (100 units each)
- Dropout layers (0.5) for regularization
- Window size: 100 days

### RNN Model  
- 4-layer LSTM architecture (50 units each)
- Dropout layers (0.2) between LSTM layers
- MinMaxScaler normalization

### XGBoost Model
- Grid search optimization for hyperparameters
- Features: 5 technical indicators with 3-day windows
- Cross-validation with 5 folds

## File Naming Conventions

- Prediction outputs: `{STOCK_NAME}_Predictions.csv`
- Model training uses train/validation split at 2020-01-01
- All notebooks output visualization plots for model evaluation

## Important Notes

- The project focuses on major tech stocks: TSLA, AAPL, GOOGL, AMZN, MSFT, META, NVDA
- Historical data spans 2010-2024 with 2020 split point for significant market regime change
- Genetic algorithm optimizes for Sharpe ratio maximization
- All models include volatility calculations for risk assessment