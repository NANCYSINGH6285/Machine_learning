# ============================================================
# FAOSTAT POLYNOMIAL REGRESSION
# BANANA & SWEET POTATO PRODUCTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
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

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. POLYNOMIAL REGRESSION FUNCTION
# ============================================================

def polynomial_regression_analysis(df, product, degree=2):

    print("\n")
    print("=" * 70)
    print(f"       {product.upper()} - POLYNOMIAL REGRESSION")
    print("=" * 70)


    # --------------------------------------------------------
    # Filter Product + Production
    # --------------------------------------------------------

    data = df[
        (df["Item"].astype(str).str.strip().str.lower() == product.lower()) &
        (df["Element"].astype(str).str.strip().str.lower() == "production")
    ].copy()


    # --------------------------------------------------------
    # Select Year and Value
    # --------------------------------------------------------

    data = data[["Year", "Value"]]

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    data["Value"] = pd.to_numeric(
        data["Value"],
        errors="coerce"
    )


    # Remove missing values
    data = data.dropna()


    # Sort by year
    data = data.sort_values("Year")


    print("\nNumber of observations:", len(data))

    print(
        f"Year range: {int(data['Year'].min())} "
        f"to {int(data['Year'].max())}"
    )


    # ========================================================
    # 4. X AND Y
    # ========================================================

    X = data[["Year"]]
    y = data["Value"]


    # ========================================================
    # 5. TRAIN-TEST SPLIT
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

    print("Training observations:", len(X_train))


    print("\nTesting Data:")
    print(
        f"{int(X_test['Year'].min())} "
        f"to "
        f"{int(X_test['Year'].max())}"
    )

    print("Testing observations:", len(X_test))


    # ========================================================
    # 6. CREATE POLYNOMIAL FEATURES
    # ========================================================

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )


    X_train_poly = poly.fit_transform(X_train)

    X_test_poly = poly.transform(X_test)


    # ========================================================
    # 7. CREATE LINEAR REGRESSION MODEL
    # ========================================================

    model = LinearRegression()


    # ========================================================
    # 8. TRAIN MODEL
    # ========================================================

    model.fit(
        X_train_poly,
        y_train
    )


    # ========================================================
    # 9. PREDICTIONS
    # ========================================================

    y_train_pred = model.predict(
        X_train_poly
    )

    y_test_pred = model.predict(
        X_test_poly
    )


    # ========================================================
    # 10. R²
    # ========================================================

    train_r2 = r2_score(
        y_train,
        y_train_pred
    )

    test_r2 = r2_score(
        y_test,
        y_test_pred
    )


    # ========================================================
    # 11. RMSE
    # ========================================================

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            y_train_pred
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_test_pred
        )
    )


    # ========================================================
    # 12. PRINT RESULTS
    # ========================================================

    print("\n")
    print("-" * 70)
    print("POLYNOMIAL REGRESSION RESULTS")
    print("-" * 70)

    print(f"\nPolynomial Degree: {degree}")


    print("\nTraining Performance:")
    print(f"R²   = {train_r2:.4f}")
    print(f"RMSE = {train_rmse:,.2f} tonnes")


    print("\nTesting Performance:")
    print(f"R²   = {test_r2:.4f}")
    print(f"RMSE = {test_rmse:,.2f} tonnes")


    # ========================================================
    # 13. ACTUAL VS PREDICTED TABLE
    # ========================================================

    prediction_table = pd.DataFrame({

        "Year": X_test["Year"].values,

        "Actual Production":
            y_test.values,

        "Predicted Production":
            y_test_pred

    })


    prediction_table["Error"] = (
        prediction_table["Actual Production"]
        -
        prediction_table["Predicted Production"]
    )


    print("\n")
    print("-" * 70)
    print("TESTING DATA - ACTUAL VS PREDICTED")
    print("-" * 70)

    print(
        prediction_table.to_string(index=False)
    )


    # ========================================================
    # 14. FUTURE PREDICTION
    # ========================================================

    future_year = 2030


    future_poly = poly.transform(
        pd.DataFrame({
            "Year": [future_year]
        })
    )


    future_prediction = model.predict(
        future_poly
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
    # 15. CREATE POLYNOMIAL CURVE
    # ========================================================

    X_curve = np.linspace(
        X["Year"].min(),
        future_year,
        300
    ).reshape(-1, 1)


    X_curve_poly = poly.transform(
        X_curve
    )


    y_curve = model.predict(
        X_curve_poly
    )


    # ========================================================
    # 16. PLOT GRAPH
    # ========================================================

    plt.figure(figsize=(13, 7))


    # Training data
    plt.scatter(
        X_train["Year"],
        y_train,
        marker="o",
        alpha=0.7,
        label="Training Data"
    )


    # Testing data
    plt.scatter(
        X_test["Year"],
        y_test,
        marker="s",
        alpha=0.9,
        label="Testing Data"
    )


    # Polynomial curve
    plt.plot(
        X_curve,
        y_curve,
        linewidth=2,
        label=f"Polynomial Regression (Degree {degree})"
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
    # 17. GRAPH LABELS
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
        f"FAOSTAT {product} Production - Polynomial Regression",
        fontsize=15
    )


    # ========================================================
    # 18. DISPLAY R² AND RMSE
    # ========================================================

    metrics_text = (
        f"Polynomial Degree = {degree}\n"
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


    # ========================================================
    # 19. RETURN RESULTS
    # ========================================================

    return {
        "model": model,
        "poly": poly,
        "data": data,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "future_prediction": future_prediction
    }


# ============================================================
# 20. BANANA POLYNOMIAL REGRESSION
# ============================================================

banana_results = polynomial_regression_analysis(
    df,
    "Bananas",
    degree=2
)


# ============================================================
# 21. SWEET POTATO POLYNOMIAL REGRESSION
# ============================================================

sweet_potato_results = polynomial_regression_analysis(
    df,
    "Sweet potatoes",
    degree=2
)


# ============================================================
# 22. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("              FINAL COMPARISON")
print("=" * 70)


print("\nBANANAS")
print(
    f"Test R²   : "
    f"{banana_results['test_r2']:.4f}"
)

print(
    f"Test RMSE : "
    f"{banana_results['test_rmse']:,.2f} tonnes"
)

print(
    f"2030 Prediction : "
    f"{banana_results['future_prediction']:,.2f} tonnes"
)


print("\nSWEET POTATOES")
print(
    f"Test R²   : "
    f"{sweet_potato_results['test_r2']:.4f}"
)

print(
    f"Test RMSE : "
    f"{sweet_potato_results['test_rmse']:,.2f} tonnes"
)

print(
    f"2030 Prediction : "
    f"{sweet_potato_results['future_prediction']:,.2f} tonnes"
)