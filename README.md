# House Price Prediction Project

This project predicts house prices based on various features using linear regression.

## Project Structure

```
HOUSE PRICE PREDICTION/
├── data/
│   └── HousingData.csv          # Boston housing dataset
├── src/
│   └── house_price_prediction.py # Main Python script
├── notebooks/
│   └── house_price_analysis.ipynb    # Jupyter notebook for analysis
├── visuals/
│   ├── data_visualization.png        # Data visualization output
│   ├── model_evaluation.png          # Model evaluation plot
│   ├── feature_importance.png        # Feature importance visualization
│   └── correlation_heatmap.png       # Correlation heatmap
├── models/
│   ├── house_price_model.pkl         # Trained model
│   └── scaler.pkl                   # Feature scaler
├── requirements.txt                  # Project dependencies
└── README.md                        # This file
```

## Dataset

The project uses the Boston Housing dataset which contains information about houses in the Boston area. The dataset includes the following features:

- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX: Nitric oxides concentration (parts per 10 million)
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built prior to 1940
- DIS: Weighted distances to five Boston employment centres
- RAD: Index of accessibility to radial highways
- TAX: Full-value property-tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population
- MEDV: Median value of owner-occupied homes in $1000s (target variable)

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install requirements with:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Python Script

Navigate to the `src` directory and run:
```bash
python house_price_prediction.py
```

This will:
1. Load and explore the dataset
2. Handle missing data
3. Preprocess inputs (normalization)
4. Split into train/test sets
5. Train a linear regression model
6. Evaluate predictions using MSE and R²
7. Save the trained model and scaler

### Using the Jupyter Notebook

Open Jupyter Notebook and run `notebooks/house_price_analysis.ipynb` for an interactive analysis.

## Results

The linear regression model achieves:
- Training R²: ~0.74
- Testing R²: ~0.66

This means the model explains about 66% of the variance in house prices in the test set.

## Key Insights

1. **Most Important Features**:
   - RM (Average number of rooms) - Positive correlation
   - LSTAT (Percentage of lower status population) - Negative correlation
   - DIS (Distance to employment centers) - Mixed effect

2. **Model Performance**:
   - The model performs reasonably well for a simple linear regression
   - There's room for improvement with more complex models

## License

This project is for educational purposes only.