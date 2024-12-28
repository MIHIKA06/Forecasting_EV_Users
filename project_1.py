import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(train_data):
    """
    Load and preprocess the training data
    """
    # Load the data
    df = pd.read_csv(train_data)
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    # Remove duplicates if any
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Function to detect outliers using IQR method
    def detect_outliers_iqr(column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        return outliers, lower_bound, upper_bound
    
    # Detect and handle outliers for each column
    print("\nOutlier Analysis:")
    for column in df.columns:
        outliers, lower_bound, upper_bound = detect_outliers_iqr(column)
        print(f"\n{column}:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Lower bound: {lower_bound:.3f}")
        print(f"Upper bound: {upper_bound:.3f}")
        
        # Cap outliers
        df[column] = df[column].clip(lower_bound, upper_bound)
    
    # Scale the features
    print("\nScaling features...")
    
    # Standard scaling for ev_generation
    scaler_ev = StandardScaler()
    df['ev_generation_scaled'] = scaler_ev.fit_transform(df[['ev_generation']])
    
    # Create interaction features
    print("\nCreating interaction features...")
    df['population_grid_interaction'] = df['population_density'] * df['grid_availability']
    df['income_grid_interaction'] = df['income'] * df['grid_availability']
    
    # Generate basic statistics
    print("\nBasic statistics after preprocessing:")
    print(df.describe())
    
    # Create visualization of distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return df

# Execute the preprocessing
if __name__ == "__main__":
    # Replace with your file path
    train_data = 'train_data.csv'
    processed_df = load_and_preprocess_data(train_data)
    
    # Save processed data
    processed_df.to_csv('processed_train_data.csv', index=False)
    print("\nProcessed data saved to 'processed_train_data.csv'")