# ============================================================
# FAOSTAT LINEAR REGRESSION
# BANANA & SWEET POTATO PRODUCTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# 1. FILE PATH
# ============================================================

file_path = r"C:\Users\Shivani Singh\Downloads\FOASTAT DATA.xlsx"


# ============================================================
# 2. READ EXCEL FILE
# ============================================================

df = pd.read_excel(file_path)

print("\nFile loaded successfully!")

print("\nColumns in dataset:")
print(df.columns.tolist())


# ============================================================
# 3. FUNCTION FOR LINEAR REGRESSION
# ============================================================

def linear_regression_analysis(df, product):

    print("\n")
    print("=" * 70)
    print(f"        {product.upper()} - LINEAR REGRESSION")
    print("=" * 70)


    # --------------------------------------------------------
    # Filter product + Production
    # --------------------------------------------------------

    data = df[
        (df["Item"].str.strip().str.lower() == product.lower()) &
        (df["Element"].str.strip().str.lower() == "production")
    ].copy()


    # --------------------------------------------------------
    # Select Year and Value
    # --------------------------------------------------------

    data = data[["Year", "Value"]]


    # Convert to numeric
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")


    # Remove missing values
    data = data.dropna()


    # Sort by year
    data = data.sort_values("Year")


    # --------------------------------------------------------
    # Display data information
    # --------------------------------------------------------

    print("\nNumber of observations:", len(data))

    print(
        f"Year range: {int(data['Year'].min())} "
        f"to {int(data['Year'].max())}"
    )


    print("\nFirst 5 production records:")
    print(data.head())


    print("\nLast 5 production records:")
    print(data.tail())


    # --------------------------------------------------------
    # X = Year
    # Y = Production
    # --------------------------------------------------------

    X = data[["Year"]]
    y = data["Value"]


    # ========================================================
    # 4. TRAIN-TEST SPLIT
    # ========================================================

    # 80% older years = Training
    # 20% recent years = Testing

    split_index = int(len(data) * 0.80)


    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]


    print("\nTraining Data:")
    print(
        f"{int(X_train['Year'].min())} "
        f"to "
        f"{int(X_train['Year'].max())}"
    )

    print("Number of training observations:", len(X_train))


    print("\nTesting Data:")
    print(
        f"{int(X_test['Year'].min())} "
        f"to "
        f"{int(X_test['Year'].max())}"
    )

    print("Number of testing observations:", len(X_test))


    # ========================================================
    # 5. CREATE LINEAR REGRESSION MODEL
    # ========================================================

    model = LinearRegression()


    # ========================================================
    # 6. TRAIN MODEL
    # ========================================================

    model.fit(X_train, y_train)


    # ========================================================
    # 7. PREDICTIONS
    # ========================================================

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)


    # ========================================================
    # 8. R² SCORE
    # ========================================================

    train_r2 = r2_score(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)


    # ========================================================
    # 9. RMSE
    # ========================================================

    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )


    # ========================================================
    # 10. REGRESSION EQUATION
    # ========================================================

    slope = model.coef_[0]

    intercept = model.intercept_


    print("\n")
    print("-" * 70)
    print("REGRESSION EQUATION")
    print("-" * 70)

    print(
        f"Production = {intercept:.2f} "
        f"+ ({slope:.2f} × Year)"
    )


    # ========================================================
    # 11. MODEL PERFORMANCE
    # ========================================================

    print("\n")
    print("-" * 70)
    print("MODEL PERFORMANCE")
    print("-" * 70)

    print("\nTraining Data:")
    print(f"R²   = {train_r2:.4f}")
    print(f"RMSE = {train_rmse:,.2f} tonnes")


    print("\nTesting Data:")
    print(f"R²   = {test_r2:.4f}")
    print(f"RMSE = {test_rmse:,.2f} tonnes")


    # ========================================================
    # 12. CREATE PREDICTION TABLE
    # ========================================================

    prediction_table = pd.DataFrame({
        "Year": X_test["Year"].values,
        "Actual Production": y_test.values,
        "Predicted Production": y_test_pred
    })


    prediction_table["Error"] = (
        prediction_table["Actual Production"]
        - prediction_table["Predicted Production"]
    )


    print("\n")
    print("-" * 70)
    print("TESTING DATA - ACTUAL VS PREDICTED")
    print("-" * 70)

    print(prediction_table.to_string(index=False))


    # ========================================================
    # 13. FUTURE PREDICTION
    # ========================================================

    future_year = 2030

    future_prediction = model.predict(
        pd.DataFrame({"Year": [future_year]})
    )[0]


    print("\n")
    print("-" * 70)
    print("FUTURE PREDICTION")
    print("-" * 70)

    print(
        f"Predicted {product} production in {future_year}: "
        f"{future_prediction:,.2f} tonnes"
    )


    # ========================================================
    # 14. REGRESSION LINE
    # ========================================================

    X_line = np.linspace(
        X["Year"].min(),
        future_year,
        200
    ).reshape(-1, 1)


    y_line = model.predict(X_line)


    # ========================================================
    # 15. PLOT
    # ========================================================

    plt.figure(figsize=(13, 7))


    # Training points
    plt.scatter(
        X_train["Year"],
        y_train,
        label="Training Data",
        marker="o",
        alpha=0.7
    )


    # Testing points
    plt.scatter(
        X_test["Year"],
        y_test,
        label="Testing Data",
        marker="s",
        alpha=0.9
    )


    # Regression line
    plt.plot(
        X_line,
        y_line,
        linewidth=2,
        label="Linear Regression Line"
    )


    # Future prediction
    plt.scatter(
        future_year,
        future_prediction,
        marker="*",
        s=250,
        label=f"2030 Prediction"
    )


    # ========================================================
    # 16. GRAPH LABELS
    # ========================================================

    plt.xlabel(
        "Year",
        fontsize=12
    )

    plt.ylabel(
        "Production (tonnes)",
        fontsize=12
    )

    plt.title(
        f"FAOSTAT {product} Production - Linear Regression",
        fontsize=15
    )


    # ========================================================
    # 17. SHOW R² AND RMSE ON GRAPH
    # ========================================================

    metrics_text = (
        f"Test R² = {test_r2:.4f}\n"
        f"Test RMSE = {test_rmse:,.0f} tonnes"
    )


    plt.text(
        0.03,
        0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8
        )
    )


    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


    # Return useful results
    return {
        "model": model,
        "data": data,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "future_prediction": future_prediction
    }


# ============================================================
# 18. RUN ANALYSIS FOR BANANAS
# ============================================================

banana_results = linear_regression_analysis(
    df,
    "Bananas"
)


# ============================================================
# 19. RUN ANALYSIS FOR SWEET POTATOES
# ============================================================

sweet_potato_results = linear_regression_analysis(
    df,
    "Sweet potatoes"
)


# ============================================================
# 20. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("             FINAL MODEL COMPARISON")
print("=" * 70)


print("\nBananas:")
print(f"Test R²   : {banana_results['test_r2']:.4f}")
print(f"Test RMSE : {banana_results['test_rmse']:,.2f} tonnes")
print(
    f"2030 Prediction : "
    f"{banana_results['future_prediction']:,.2f} tonnes"
)


print("\nSweet Potatoes:")
print(f"Test R²   : {sweet_potato_results['test_r2']:.4f}")
print(f"Test RMSE : {sweet_potato_results['test_rmse']:,.2f} tonnes")
print(
    f"2030 Prediction : "
    f"{sweet_potato_results['future_prediction']:,.2f} tonnes"
)