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

        # Evaluate models on the validation set
    print("Evaluation Results (Validation Set):")
    evaluate_model(linear_model, "Linear Regression", x_val, y_val)
    evaluate_model(ridge_model, "Ridge Regression", x_val_scaled, y_val)
    evaluate_model(lasso_model, "Lasso Regression", x_val_scaled, y_val)
    evaluate_model(logistic_model, "Logistic Regression", x_val, y_val)
    
    # Evaluate models on the test set
    print("Final Evaluation Results (Test Set):")
    evaluate_model(linear_model, "Linear Regression", x_test, y_test)
    evaluate_model(ridge_model, "Ridge Regression", x_test_scaled, y_test)
    evaluate_model(lasso_model, "Lasso Regression", x_test_scaled, y_test)
    evaluate_model(logistic_model, "Logistic Regression", x_test, y_test)
    
    # Calculate AUC-ROC for Logistic Regression on the validation set
    y_pred_logistic_proba_val = logistic_model.predict_proba(x_val)[:, 1] 
    auc_val = roc_auc_score(y_val, y_pred_logistic_proba_val)
    print(f"Logistic Regression AUC-ROC on Validation Set: {auc_val:.4f}\n")
    
    # Print R² and accuracy scores for all models on train and test sets
    linear_r2_train = linear_model.rsquared
    linear_r2_test = linear_model.rsquared 
    ridge_score_train = ridge_model.score(x_train_scaled, y_train)
    ridge_score_test = ridge_model.score(x_test_scaled, y_test)
    lasso_score_train = lasso_model.score(x_train_scaled, y_train)
    lasso_score_test = lasso_model.score(x_test_scaled, y_test)
    logistic_score_train = logistic_model.score(x_train, y_train) * 100
    logistic_score_test = logistic_model.score(x_test, y_test) * 100
    
    print(f"Linear Regression R² on Train: {linear_r2_train:.4f}")
    print(f"Linear Regression R² on Test: {linear_r2_test:.4f}")
    print(f"Ridge Regression R² on Train: {ridge_score_train:.4f}")
    print(f"Ridge Regression R² on Test: {ridge_score_test:.4f}")
    print(f"Lasso Regression R² on Train: {lasso_score_train:.4f}")
    print(f"Lasso Regression R² on Test: {lasso_score_test:.4f}")
    print(f"Logistic Regression Accuracy on Train: {logistic_score_train:.2f}%")
    print(f"Logistic Regression Accuracy on Test: {logistic_score_test:.2f}%")

    # Check model assumptions for each model
    result_linear = check_model_assumptions(linear_model, x_train, y_train)
    print(result_linear)

    result_ridge = check_model_assumptions(ridge_model, x_train, y_train)
    print(result_ridge)

    result_lasso = check_model_assumptions(lasso_model, x_train, y_train)
    print(result_lasso)

    result_logistic = check_model_assumptions(logistic_model, x_train, y_train)
    print(result_logistic)

    # Calculate Variance Inflation Factor (VIF)
    vif_result = calculate_vif(x_train)
    print(vif_result)

    pcos_visualization(df, logistic_model, x_test, y_test)

   # Example of user input (replace with dynamic input if needed)
    user_input = [4, 2.5, 0.8, 1, 35.5, 7, 1, 1, 1, 1, 25]
    # Order: ["Cycle(R/I)", "AMH(ng/mL)", "PRG(ng/mL)", "Fast food(Y/N)", "BMI", "Cycle length(days)",
    # "Weight gain(Y/N)", "Hair growth(Y/N)", "Skin darkening(Y/N)", "Pimples(Y/N)", "Follicle No.(L)"]

    # Scale the user input using the fitted scaler
    user_input_scaled = scaler.transform([user_input])

    # Predict the risk using the logistic model
    logistic_risk = logistic_model.predict(user_input_scaled)

    # Print the risk prediction
    print("\nRisk prediction using logistic regression:")
    if logistic_risk[0] == 1:
        print("Prediction: PCOS is potentially present. Please consult a medical professional for further evaluation.")
    else:
        print("Prediction: PCOS is unlikely to be present. However, if you have concerns, consider consulting a medical professional.")

if __name__ == "__main__":
    main()
    