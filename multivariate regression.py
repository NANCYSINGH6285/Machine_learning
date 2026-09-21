# ============================================================
# FAOSTAT MULTIVARIATE LINEAR REGRESSION
# BANANA & SWEET POTATO PRODUCTION
#
# Features:
#   1. Year
#   2. Area harvested
#   3. Yield
#
# Target:
#   Production
#
# Includes:
#   Training data
#   Testing data
#   Prediction
#   R²
#   RMSE
#   Actual vs Predicted graph
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
# 3. FUNCTION FOR MULTIVARIATE REGRESSION
# ============================================================

def multivariate_regression_analysis(df, product):

    print("\n")
    print("=" * 75)
    print(f"       {product.upper()} - MULTIVARIATE REGRESSION")
    print("=" * 75)


    # ========================================================
    # 4. FILTER DATA FOR SELECTED PRODUCT
    # ========================================================

    product_data = df[
        df["Item"].astype(str).str.strip().str.lower()
        == product.lower()
    ].copy()


    # ========================================================
    # 5. SEPARATE AREA, YIELD AND PRODUCTION
    # ========================================================

    area_data = product_data[
        product_data["Element"].astype(str).str.strip().str.lower()
        == "area harvested"
    ][["Year", "Value"]].copy()

    yield_data = product_data[
        product_data["Element"].astype(str).str.strip().str.lower()
        == "yield"
    ][["Year", "Value"]].copy()

    production_data = product_data[
        product_data["Element"].astype(str).str.strip().str.lower()
        == "production"
    ][["Year", "Value"]].copy()


    # ========================================================
    # 6. RENAME COLUMNS
    # ========================================================

    area_data.rename(
        columns={"Value": "Area_Harvested"},
        inplace=True
    )

    yield_data.rename(
        columns={"Value": "Yield"},
        inplace=True
    )

    production_data.rename(
        columns={"Value": "Production"},
        inplace=True
    )


    # ========================================================
    # 7. MERGE ALL THREE DATASETS USING YEAR
    # ========================================================

    data = pd.merge(
        area_data,
        yield_data,
        on="Year",
        how="inner"
    )

    data = pd.merge(
        data,
        production_data,
        on="Year",
        how="inner"
    )


    # ========================================================
    # 8. CLEAN DATA
    # ========================================================

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    data["Area_Harvested"] = pd.to_numeric(
        data["Area_Harvested"],
        errors="coerce"
    )

    data["Yield"] = pd.to_numeric(
        data["Yield"],
        errors="coerce"
    )

    data["Production"] = pd.to_numeric(
        data["Production"],
        errors="coerce"
    )


    data = data.dropna()

    data = data.sort_values("Year")


    print("\nNumber of observations:", len(data))

    print("\nPrepared Data:")
    print(data.head())


    # ========================================================
    # 9. DEFINE FEATURES AND TARGET
    # ========================================================

    # Multiple independent variables
    X = data[
        [
            "Year",
            "Area_Harvested",
            "Yield"
        ]
    ]

    # Dependent variable
    y = data["Production"]


    # ========================================================
    # 10. TRAIN-TEST SPLIT
    # ========================================================

    # 80% older years = Training
    # 20% recent years = Testing

    split_index = int(len(data) * 0.80)


    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]


    print("\n")
    print("-" * 75)
    print("TRAINING AND TESTING DATA")
    print("-" * 75)


    print(
        "\nTraining years:",
        int(data.iloc[:split_index]["Year"].min()),
        "to",
        int(data.iloc[:split_index]["Year"].max())
    )

    print(
        "Training observations:",
        len(X_train)
    )


    print(
        "\nTesting years:",
        int(data.iloc[split_index:]["Year"].min()),
        "to",
        int(data.iloc[split_index:]["Year"].max())
    )

    print(
        "Testing observations:",
        len(X_test)
    )


    # ========================================================
    # 11. CREATE MULTIVARIATE LINEAR REGRESSION MODEL
    # ========================================================

    model = LinearRegression()


    # ========================================================
    # 12. TRAIN MODEL
    # ========================================================

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # 13. PREDICTIONS
    # ========================================================

    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )


    # ========================================================
    # 14. R² SCORE
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
    # 15. RMSE
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
    # 16. DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 75)
    print("MODEL PERFORMANCE")
    print("=" * 75)


    print("\nTraining Performance:")
    print(
        f"R²   = {train_r2:.4f}"
    )

    print(
        f"RMSE = {train_rmse:,.2f} tonnes"
    )


    print("\nTesting Performance:")
    print(
        f"R²   = {test_r2:.4f}"
    )

    print(
        f"RMSE = {test_rmse:,.2f} tonnes"
    )


    # ========================================================
    # 17. REGRESSION COEFFICIENTS
    # ========================================================

    print("\n")
    print("=" * 75)
    print("REGRESSION COEFFICIENTS")
    print("=" * 75)


    print(
        f"Intercept = {model.intercept_:,.2f}"
    )

    print(
        f"Year coefficient = "
        f"{model.coef_[0]:,.4f}"
    )

    print(
        f"Area Harvested coefficient = "
        f"{model.coef_[1]:,.4f}"
    )

    print(
        f"Yield coefficient = "
        f"{model.coef_[2]:,.4f}"
    )


    # ========================================================
    # 18. ACTUAL VS PREDICTED TABLE
    # ========================================================

    prediction_table = pd.DataFrame({

        "Year":
            X_test["Year"].values,

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
    print("=" * 75)
    print("TESTING DATA - ACTUAL VS PREDICTED")
    print("=" * 75)

    print(
        prediction_table.to_string(index=False)
    )


    # ========================================================
    # 19. FUTURE PREDICTION
    # ========================================================

    # NOTE:
    # For future production prediction, we need
    # future Year + future Area + future Yield.
    #
    # Therefore we use the latest available Area and Yield
    # as an example scenario for 2030.

    latest_area = data.iloc[-1]["Area_Harvested"]

    latest_yield = data.iloc[-1]["Yield"]

    future_year = 2030


    future_data = pd.DataFrame({

        "Year": [future_year],

        "Area_Harvested": [latest_area],

        "Yield": [latest_yield]

    })


    future_prediction = model.predict(
        future_data
    )[0]


    print("\n")
    print("=" * 75)
    print("2030 PREDICTION")
    print("=" * 75)


    print(
        f"Year = {future_year}"
    )

    print(
        f"Area Harvested = "
        f"{latest_area:,.2f}"
    )

    print(
        f"Yield = "
        f"{latest_yield:,.2f}"
    )

    print(
        f"\nPredicted {product} Production = "
        f"{future_prediction:,.2f} tonnes"
    )


    # ========================================================
    # 20. GRAPH 1
    # ACTUAL VS PREDICTED
    # ========================================================

    plt.figure(figsize=(12, 7))


    plt.scatter(
        y_train,
        y_train_pred,
        marker="o",
        alpha=0.7,
        label="Training Data"
    )


    plt.scatter(
        y_test,
        y_test_pred,
        marker="s",
        alpha=0.9,
        label="Testing Data"
    )


    # Perfect prediction line
    minimum = min(
        y.min(),
        y_train_pred.min(),
        y_test_pred.min()
    )

    maximum = max(
        y.max(),
        y_train_pred.max(),
        y_test_pred.max()
    )


    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linewidth=2,
        label="Perfect Prediction"
    )


    # Future prediction
    plt.scatter(
        y_test.max(),
        y_test_pred.max(),
        marker="*",
        s=200,
        label="Prediction"
    )


    plt.xlabel(
        "Actual Production (tonnes)",
        fontsize=12
    )

    plt.ylabel(
        "Predicted Production (tonnes)",
        fontsize=12
    )

    plt.title(
        f"{product} - Multivariate Regression\n"
        "Actual vs Predicted Production",
        fontsize=15
    )


    # Metrics
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

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


    # ========================================================
    # 21. GRAPH 2
    # TESTING DATA - ACTUAL VS PREDICTED OVER YEARS
    # ========================================================

    plt.figure(figsize=(13, 7))


    plt.plot(
        X_test["Year"],
        y_test,
        marker="o",
        linewidth=2,
        label="Actual Production"
    )


    plt.plot(
        X_test["Year"],
        y_test_pred,
        marker="s",
        linewidth=2,
        label="Predicted Production"
    )


    plt.xlabel(
        "Year",
        fontsize=12
    )

    plt.ylabel(
        "Production (tonnes)",
        fontsize=12
    )

    plt.title(
        f"{product} - Testing Data\n"
        "Actual vs Predicted Production",
        fontsize=15
    )


    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


    # ========================================================
    # 22. RETURN RESULTS
    # ========================================================

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
# 23. BANANA MODEL
# ============================================================

banana_results = multivariate_regression_analysis(
    df,
    "Bananas"
)


# ============================================================
# 24. SWEET POTATO MODEL
# ============================================================

sweet_potato_results = multivariate_regression_analysis(
    df,
    "Sweet potatoes"
)


# ============================================================
# 25. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 75)
print("                 FINAL COMPARISON")
print("=" * 75)


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