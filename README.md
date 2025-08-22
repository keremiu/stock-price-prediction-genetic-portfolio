# Stock Price Prediction & Genetic Portfolio Optimization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive computational finance project that combines advanced machine learning models for stock price prediction with genetic algorithms for portfolio optimization. This project uses multiple deep learning approaches and statistical analysis to create optimal investment portfolios.

## 🎯 Project Overview

This project implements three different machine learning models to predict stock prices and uses genetic algorithms to optimize portfolio allocation based on risk-return characteristics. The system analyzes major tech stocks (AAPL, GOOGL, TSLA, AMZN, MSFT, META, NVDA) and generates optimized portfolios with minimized risk and maximized returns.

**Academic Context:** YAP471 - Computational Finance Project, Spring 2023-2024, TOBB University of Economics and Technology

## ✨ Key Features

### 🤖 Machine Learning Models
- **CNN-LSTM Hybrid**: Combines Convolutional Neural Networks with Long Short-Term Memory networks
- **RNN (Recurrent Neural Network)**: Multi-layer LSTM architecture for time series prediction
- **XGBoost**: Gradient boosting with technical indicators and grid search optimization

### 📊 Technical Analysis
- **Moving Averages**: 20, 50, 100, 200-day periods
- **MACD** (Moving Average Convergence Divergence)
- **RSI** (Relative Strength Index)
- **Bollinger Bands**: Upper, middle, lower bands
- **Ichimoku Cloud**: Complete Ichimoku indicator system

### 🧬 Genetic Algorithm Portfolio Optimization
- **Population-based optimization** with configurable parameters
- **Sharpe ratio maximization** for optimal risk-adjusted returns
- **Multi-point crossover** and adaptive mutation strategies
- **Elitism preservation** of best-performing individuals

### 📈 Risk Analysis
- **Volatility calculations** using log returns
- **Portfolio risk assessment** through covariance matrices
- **Performance metrics**: R² score, MAE, MSE, explained variance

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install pandas numpy yfinance tensorflow xgboost scikit-learn matplotlib seaborn ta tqdm keras
```

### Clone Repository
```bash
git clone https://github.com/yourusername/stock-price-prediction-genetic-portfolio.git
cd stock-price-prediction-genetic-portfolio
```

## 🚀 Usage

### 1. Stock Price Prediction

#### Individual Model Training
Navigate to the `Stock-Price-Prediction/` directory and run any of the following notebooks:

- `CNN_LSTM_Stock_Price_Prediction.ipynb` - Hybrid CNN-LSTM model
- `RNN_Stock_Price_Prediction.ipynb` - Recurrent neural network approach  
- `xgboost-stock-price-prediction.ipynb` - XGBoost regression

#### Multi-Model Comparison
Use `Stock-Price-Prediction-Optimal-Portfolio.ipynb` to run all three models on the same dataset:

```python
# Set the stock symbol
STOCK_NAME = "AAPL"  # or "GOOGL", "TSLA", etc.

# Models will automatically:
# 1. Download historical data (2010-2024)
# 2. Apply technical indicators
# 3. Train all three models
# 4. Generate predictions with volatility analysis
```

### 2. Portfolio Optimization

Run the genetic algorithm section in `finance_genetic.ipynb`:

```python
# Genetic Algorithm Parameters
population_size = 1000
num_generations = 90
mutation_rate = 0.10

# The algorithm will optimize portfolio weights to maximize Sharpe ratio
```

### 3. Exploratory Data Analysis

Check `Explanatory-Data-Analysis/` for comprehensive data analysis and visualization notebooks.

## 🏗️ Project Architecture

```
Risk-Analysis-and-Portfolio-Creation-with-Genetic-Algorithms-using-statiscal-analysis/
├── Stock-Price-Prediction/
│   ├── CNN_LSTM_Stock_Price_Prediction.ipynb
│   ├── RNN_Stock_Price_Prediction.ipynb
│   └── xgboost-stock-price-prediction.ipynb
├── Optimal-Portfolio-Creation/
│   └── finance_genetic.ipynb
├── Explanatory-Data-Analysis/
│   └── Explanatory Data Analysis.ipynb
├── Prediction_Datasets/
│   ├── AAPL_Predictions.csv
│   ├── GOOGL_Predictions.csv
│   └── [other stock predictions]
├── Stock-Price-Prediction-Optimal-Portfolio.ipynb
├── finance_genetic.ipynb
└── README.md
```

## 🧠 Model Architectures

### CNN-LSTM Hybrid Model
- **CNN Layers**: 3 TimeDistributed Conv1D layers (64→128→64 filters)
- **LSTM Layers**: 2 Bidirectional LSTM layers (100 units each)
- **Regularization**: Dropout layers (0.5) and MaxPooling
- **Window Size**: 100 days for sequence learning
- **Input Shape**: (batch_size, 1, 100, 1)

### RNN Model
- **Architecture**: 4-layer LSTM network (50 units per layer)
- **Dropout**: 0.2 between each LSTM layer
- **Normalization**: MinMaxScaler for feature scaling
- **Return Sequences**: True for first 3 layers, False for last layer

### XGBoost Model
- **Features**: 5 technical indicators with 3-day sliding windows
- **Optimization**: GridSearchCV with 5-fold cross-validation
- **Hyperparameters**: n_estimators, learning_rate, max_depth, gamma
- **Objective**: Regression with squared error loss

## 📊 Results & Performance

### Model Performance Metrics
| Model | Variance Score | R² Score | Max Error | Training Time |
|-------|---------------|----------|-----------|---------------|
| CNN-LSTM | ~0.81 | ~0.81 | ~0.37 | ~100 epochs |
| RNN | ~0.76 | ~0.76 | ~0.34 | ~25 epochs |
| XGBoost | ~0.95 | ~0.95 | ~0.06 | Grid Search |

### Portfolio Optimization Results
- **Optimized Volatility**: ~0.298 (29.8% annual risk)
- **Asset Allocation**: Diversified across 7 major tech stocks
- **Optimization Method**: Sharpe ratio maximization
- **Population Evolution**: 90 generations with 1000 individuals

## 🔧 Configuration

### Genetic Algorithm Parameters
```python
# Population and Evolution
population_size = 1000        # Number of individuals per generation
num_generations = 90          # Evolution iterations
mutation_rate = 0.10         # Probability of mutation (10-20%)

# Portfolio Constraints
num_assets = 7               # Number of stocks in portfolio
risk_free_rate = 0.02        # Assumed risk-free rate for Sharpe calculation
```

### Data Parameters
```python
# Historical Data Range
start_date = "2010-01-01"
end_date = "2024-01-01"
train_test_split = "2020-01-01"  # COVID-19 regime change

# Technical Indicators
moving_averages = [20, 50, 100, 200]
rsi_period = 14
macd_periods = (12, 26, 9)
bollinger_period = 20
```

## 📈 Supported Assets

The system currently supports analysis of major technology stocks:
- **TSLA** - Tesla, Inc.
- **AAPL** - Apple Inc.
- **GOOGL** - Alphabet Inc.
- **AMZN** - Amazon.com Inc.
- **MSFT** - Microsoft Corporation
- **META** - Meta Platforms Inc.
- **NVDA** - NVIDIA Corporation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors & Contributors

- **[Kerem Ihsan Ulasan](https://github.com/keremiu)** - Lead Developer
- **[Omer Faruk Merey](https://github.com/OmerFarukMerey)** - Co-Developer

## 🎓 Academic Information

- **Course**: YAP471 - Computational Finance
- **Term**: Spring 2023-2024
- **Institution**: TOBB University of Economics and Technology
- **Topic**: Risk Analysis and Portfolio Creation with Genetic Algorithms using Statistical Analysis

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you have any questions or issues, please open an issue on GitHub or contact the authors directly.

## 🙏 Acknowledgments

- Yahoo Finance API for historical stock data
- TensorFlow and Keras teams for deep learning frameworks
- XGBoost developers for gradient boosting implementation
- TA-Lib community for technical analysis indicators
- TOBB University for academic support and guidance

---

**⚠️ Disclaimer**: This project is for educational and research purposes only. Do not use the predictions for actual trading decisions without proper risk assessment and financial advice.