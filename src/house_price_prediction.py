import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the dataset and explore data distributions
def load_and_explore_data():
    """
    Load the housing dataset and perform initial exploration
    """
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv('../data/HousingData.csv')
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nStatistical summary:")
    print(df.describe())
    
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    return df

# Step 2: Handle missing data and preprocess inputs
def preprocess_data(df):
    """
    Handle missing data and preprocess the dataset
    """
    print("\nPreprocessing data...")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values[missing_values > 0])
    
    # Handle missing values by filling with median (more robust than mean)
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            df[column].fillna(df[column].median(), inplace=True)
    
    print("\nMissing values after imputation:")
    print(df.isnull().sum())
    
    # Separate features and target variable
    # MEDV is the target variable (house prices)
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

# Step 3: Data visualization
def visualize_data(X, y):
    """
    Visualize data distributions and relationships
    """
    print("\nCreating visualizations...")
    
    # Plot target variable distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=30, edgecolor='black')
    plt.title('Distribution of House Prices (MEDV)')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    # Plot target variable distribution
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=30, edgecolor='black')
    plt.title('Distribution of House Prices (MEDV)')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    # Correlation heatmap
    plt.subplot(1, 2, 2)
    correlation_matrix = pd.concat([X, y], axis=1).corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../visuals/data_visualization.png')
    plt.close()
    
    # Save correlation heatmap separately
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../visuals/correlation_heatmap.png')
    plt.close()
    
    print("Visualizations saved to visuals/data_visualization.png and visuals/correlation_heatmap.png")

# Step 4: Feature normalization
def normalize_features(X_train, X_test):
    """
    Normalize features using StandardScaler
    """
    print("\nNormalizing features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler

# Step 5: Split into train/test sets
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    print(f"\nSplitting data into train/test sets ({1-test_size}:{test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

# Step 6: Train a regression model
def train_model(X_train, y_train):
    """
    Train a linear regression model
    """
    print("\nTraining Linear Regression model...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    return model

# Step 7: Evaluate predictions
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the model using various metrics
    """
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    
    # Plot predictions vs actual values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Testing Set: Actual vs Predicted\nR² = {test_r2:.4f}')
    
    plt.tight_layout()
    plt.savefig('../visuals/model_evaluation.png')
    plt.close()
    
    print("Evaluation plot saved to visuals/model_evaluation.png")
    
    return y_test_pred

# Step 8: Feature importance analysis
def analyze_feature_importance(model, X):
    """
    Analyze and visualize feature importance
    """
    print("\nAnalyzing feature importance...")
    
    # Get feature coefficients
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    })
    
    # Sort by absolute coefficient value
    feature_importance['abs_coefficient'] = np.abs(feature_importance['coefficient'])
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='abs_coefficient', y='feature')
    plt.title('Top 10 Feature Importance (Absolute Coefficients)')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig('../visuals/feature_importance.png')
    plt.close()
    
    print("Feature importance plot saved to visuals/feature_importance.png")

# Main function to run the complete pipeline
def main():
    """
    Main function to execute the complete house price prediction pipeline
    """
    print("="*60)
    print("HOUSE PRICE PREDICTION PROJECT")
    print("="*60)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    
    # Step 2: Preprocess data
    X, y = preprocess_data(df)
    
    # Step 3: Visualize data
    visualize_data(X, y)
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Step 6: Train model
    model = train_model(X_train_scaled, y_train)
    
    # Step 7: Evaluate model
    y_test_pred = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 8: Analyze feature importance
    analyze_feature_importance(model, X)
    
    # Save the trained model
    import joblib
    joblib.dump(model, '../models/house_price_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    print("\nModel and scaler saved to models/ directory")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()