import warnings
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from helper import *

def main():
    # Suppress warnings for a cleaner output
    warnings.filterwarnings("ignore")
    
    # Load dataset
    df = pd.read_excel("data/data.xlsx", sheet_name="Full_new")

    # Define columns for categorical data visualization
    columns = ["PCOS(Y/N)", "Pregnant(Y/N)", "Weight gain(Y/N)", "Hair growth(Y/N)", "Skin darkening(Y/N)", "Hair loss(Y/N)", 
               "Pimples(Y/N)", "Fast food(Y/N)", "Reg.Exercise(Y/N)"]
    
    # Create bar plots for categorical columns
    for column in columns:
        bar_plot(df, column)

    # Define columns for numerical data visualization
    columns = ["Age(yrs)", "Weight(kg)", "Height(cm)", "Cycle length(days)"]
    
    # Create histograms for numerical columns
    for column in columns:
        plot_histogram(df, column)

    # Plot BMI histogram and blood group histogram
    bmi_histogram(df)
    blood_group_histogram(df)

    # Check for missing values in the dataset
    print(check_for_missing_values(df))

    # Interpolate missing values for selected columns using linear and spline methods
    df["Age(yrs)"] = df["Age(yrs)"].interpolate(method="linear", limit_direction="both")
    df["Pulse rate(bpm)"] = df["Pulse rate(bpm)"].interpolate(method="spline", order=4, limit_direction="both")
    df["Marraige Status(Yrs)"] = df["Marraige Status(Yrs)"].interpolate(method="spline", order=4, limit_direction="both")
    df["AMH(ng/mL)"] = df["AMH(ng/mL)"].interpolate(method="spline", order=4, limit_direction="both")
    df["Vit D3(ng/mL)"] = df["Vit D3(ng/mL)"].interpolate(method="spline", order=4, limit_direction="both")

    # Fill missing values for "Fast food(Y/N)" column with median value
    df["Fast food(Y/N)"] = df["Fast food(Y/N)"].fillna(df['Fast food(Y/N)'].median())

    # Make a copy of the dataframe and drop non-relevant columns, show correlation matrix
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=["Sl. No", "Patient File No."])
    correlation_matrix(df_copy)
    
    # Split data into features (X) and target (Y)
    Y = df["PCOS(Y/N)"]
    X = df[["Cycle(R/I)", "AMH(ng/mL)", "PRG(ng/mL)", "Fast food(Y/N)", "BMI", "Cycle length(days)",
            "Weight gain(Y/N)", "Hair growth(Y/N)", "Skin darkening(Y/N)", 
            "Pimples(Y/N)", "Follicle No.(L)"]]
 
    # Split data into training, validation, and test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.2, random_state=60)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=60)
    
    # Standardize the features for training, validation, and test sets
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    
    # Fit models using different regression techniques
    linear_model = get_linear_model(x_train, y_train)
    ridge_model = get_ridge_model(x_train_scaled, y_train)
    lasso_model = get_lasso_model(x_train_scaled, y_train)
    logistic_model = get_logistic_model(x_train, y_train)
    