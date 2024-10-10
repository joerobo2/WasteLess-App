import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle

# Banner Placeholder
st.image('Logo.png', use_column_width=True)  # Placeholder for the banner above the title

# Title and Application Explainer
st.title("WasteLess")
st.write("""
Welcome to WasteLess - a smart food waste prediction and recommendation system! 
This app helps households minimize food waste by predicting potential waste and providing actionable insights. 
Using machine learning models, we generate personalized recommendations to help you reduce waste, save money, and contribute to sustainability.
""")

# Load dataset automatically
data = pd.read_csv("food_data.csv")

# Sidebar user input
st.sidebar.header("User Input")
family_size = st.sidebar.slider("Family Size", 1, 10, 3)
purchase_amount = st.sidebar.number_input("Purchase Amount (kg)", min_value=0.0, value=10.0)
consumption_amount = st.sidebar.number_input("Consumption Amount (kg)", min_value=0.0, value=8.0)
dietary_preferences = st.sidebar.selectbox("Dietary Preferences",
                                           ["Omnivore", "Vegetarian", "Vegan", "Pescatarian", "Meat"])


# Preprocess the data
def preprocess_data(df, is_prediction_input=False, encoder=None, scaler=None):
    if not is_prediction_input:
        y = df['Waste_Amount']  # Target variable
    else:
        y = None

    X = df.drop(columns=['Waste_Amount', 'Household_ID', 'Timestamp'], errors='ignore')

    # Interaction Features
    X['Purchase_Consumption_Interaction'] = X['Purchase_Amount'] * X['Consumption_Amount']

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X[['Purchase_Amount', 'Consumption_Amount', 'Family_Size']])
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(
        ['Purchase_Amount', 'Consumption_Amount', 'Family_Size']))

    # Drop original columns to avoid duplication issues
    X = X.drop(columns=['Purchase_Amount', 'Consumption_Amount', 'Family_Size'], errors='ignore')

    # Concatenate polynomial features
    X = pd.concat([X.reset_index(drop=True), X_poly_df.reset_index(drop=True)], axis=1)

    # One-hot encoding for 'Dietary_Preferences'
    if encoder is None:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        dietary_encoded = encoder.fit_transform(df[['Dietary_Preferences']])
    else:
        dietary_encoded = encoder.transform(df[['Dietary_Preferences']])
    dietary_encoded_df = pd.DataFrame(dietary_encoded, columns=encoder.get_feature_names_out(['Dietary_Preferences']))
    X = pd.concat([X.reset_index(drop=True), dietary_encoded_df.reset_index(drop=True)], axis=1)

    # Feature Scaling
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    else:
        X_scaled = scaler.transform(X.select_dtypes(include=[np.number]))
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns)

    return X_scaled_df, y, encoder, scaler


# Preprocess the data
X, y, encoder, scaler = preprocess_data(data)

# Save encoder and scaler for later use
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Models to evaluate
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'SVM': SVR(kernel='rbf', C=1.0, epsilon=0.1)
}


# Function to evaluate a model
def evaluate_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred)

    return {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-Squared': r2, 'Predictions': y_pred
    }


# Create a tab layout for Recommendations, Insights, and Developer Tools
tabs = st.tabs(["Recommendations", "Insights", "Developer Tools"])

# Recommendations Tab
with tabs[0]:
    st.header("Generate Recommendations")

    # Model Selection
    model_choice = st.selectbox("Choose a model for prediction", list(models.keys()))

    # Generate Recommendations button
    if st.button("Generate Recommendations"):
        selected_model = models[model_choice]
        selected_model.fit(X, y)

        # Generate prediction for user input parameters
        input_data = pd.DataFrame({
            'Family_Size': [family_size],
            'Purchase_Amount': [purchase_amount],
            'Consumption_Amount': [consumption_amount],
            'Dietary_Preferences': [dietary_preferences]
        })

        # Load encoder and scaler
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        input_data_processed, _, _, _ = preprocess_data(
            input_data, is_prediction_input=True, encoder=encoder, scaler=scaler
        )

        prediction = selected_model.predict(input_data_processed)
        st.write(f"Predicted Waste Amount: {prediction[0]:.2f} kg")

        # Recommendations based on prediction
        st.subheader("Recommendations to Reduce Waste:")
        recommendations = []
        if prediction[0] > 5.0:
            recommendations.append("1. Consider purchasing less food to better match your consumption needs.")
        if dietary_preferences == 'Vegan' and prediction > 3.0:
            recommendations.append(
                "2. Vegan diets can have perishable items - try to store food in optimal conditions to increase shelf life.")
        if family_size > 5 and prediction > 4.0:
            recommendations.append(
                "3. With a larger family, consider meal planning to ensure all food purchased is consumed.")
        if not recommendations:
            recommendations.append("1. Your current purchasing habits are optimal. Keep up the good work!")

        for rec in recommendations:
            st.write(rec)

        # Create and display a donut chart for waste vs consumption
        waste = prediction[0]
        consumption = consumption_amount

        # Create the donut chart data
        values = [waste, consumption]
        labels = ['Predicted Waste (kg)', 'Consumption (kg)']
        colors = ['#FF9999', '#66B3FF']

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, marker=dict(colors=colors))])
        fig.update_layout(title_text='Waste vs Consumption')
        st.plotly_chart(fig)

# Insights Tab
with tabs[1]:
    st.header("Insights for Waste Reduction")
    st.write("""
    Based on the current parameters, here are some insights that could help you reduce your food waste:
    """)

    if 'waste' in locals() and prediction[0] > consumption:
        st.write(
            "- **High Waste Alert**: Your predicted waste is higher than your consumption. Consider reducing the purchase amount or improving food storage techniques.")

    if dietary_preferences in ['Vegan', 'Vegetarian']:
        st.write(
            f"- **Dietary Preference Insight**: Since you follow a {dietary_preferences.lower()} diet, you might have more perishable items. Focus on proper storage and consuming items before they spoil.")

    if family_size > 4:
        st.write(
            "- **Large Family Insight**: Larger families can benefit from bulk purchasing but should be careful about over-purchasing perishable goods. Meal prepping can help reduce waste.")

    st.write(
        "- **General Tip**: Proper storage and meal planning are key strategies for reducing waste. You can also try freezing leftovers to extend shelf life.")

    # Add charts for visual insights
    fig1 = px.histogram(data, x='Family_Size', title='Family Size Distribution', labels={'Family_Size': 'Family Size'}, color_discrete_sequence=px.colors.qualitative.Prism)
    fig1.update_layout(title_text='Family Size Distribution', title_x=0.5)
    st.plotly_chart(fig1)
    st.write("This chart shows the distribution of different family sizes in the dataset. It helps us understand how waste may vary depending on family size.")

    fig2 = px.histogram(data, x='Dietary_Preferences', title='Dietary Preferences Distribution',
                        labels={'Dietary_Preferences': 'Dietary Preferences'}, color_discrete_sequence=px.colors.qualitative.Set3)
    fig2.update_layout(title_text='Dietary Preferences Distribution', title_x=0.5)
    st.plotly_chart(fig2)
    st.write("This chart highlights the different dietary preferences within the dataset. The information helps identify how dietary choices influence food waste.")

    fig3 = px.scatter(data, x='Purchase_Amount', y='Waste_Amount', title='Purchase Amount vs Waste',
                      labels={'Purchase_Amount': 'Purchase Amount (kg)', 'Waste_Amount': 'Waste Amount (kg)'}, color_discrete_sequence=px.colors.qualitative.Safe)
    fig3.update_layout(title_text='Purchase Amount vs Waste', title_x=0.5)
    st.plotly_chart(fig3)
    st.write("This scatter plot shows the relationship between the purchase amount and waste. It can help identify trends in over-purchasing.")

    fig4 = px.pie(data, names='Dietary_Preferences', title='Dietary Preferences Breakdown', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set1)
    fig4.update_layout(title_text='Dietary Preferences Breakdown', title_x=0.5)
    st.plotly_chart(fig4)
    st.write("This donut chart represents the breakdown of dietary preferences in the dataset, providing insights into which dietary habits may contribute to higher waste.")

# Developer Tools Tab
with tabs[2]:
    st.header("Model Performance Insights")

    # Dynamic Explainer based on Model Choice
    if model_choice == "Random Forest":
        st.write("""
        **Random Forest Explainer:**
        Random Forest is an ensemble learning method that combines multiple decision trees. It's generally robust to overfitting and can handle both numerical and categorical features. Evaluate its performance using metrics like MAE, MSE, RMSE, and R-Squared.
        """)
    elif model_choice == "Linear Regression":
        st.write("""
        **Linear Regression Explainer:**
        Linear Regression models a linear relationship between the independent variables and the dependent variable. It's suitable for simple relationships but might not capture complex patterns. Evaluate its performance using metrics like MAE, MSE, RMSE, and R-Squared.
        """)
    elif model_choice == "XGBoost":
        st.write("""
        **XGBoost Explainer:**
        XGBoost is a gradient boosting framework that creates a series of decision trees, each learning from the errors of its predecessors. It's often used for its speed and accuracy, especially in large datasets. Evaluate its performance using metrics like MAE, MSE, RMSE, and R-Squared.
        """)
    elif model_choice == "SVM":
        st.write("""
        **SVM (Support Vector Machine) Explainer:**
        SVM finds a hyperplane that separates data points into different classes. It's effective for non-linear relationships but can be computationally expensive. Evaluate its performance using metrics like MAE, MSE, RMSE, and R-Squared.
        """)

    # Evaluate models and display results
    evaluation_results = {}
    for model_name, model in models.items():
        evaluation_results[model_name] = evaluate_model(model, X, y)

    # Convert results to DataFrame
    results_df = pd.DataFrame(evaluation_results).T
    results_df = results_df[['MAE', 'MSE', 'RMSE', 'R-Squared']]
    # Highlight the metrics of the selected model
    results_df_styled = results_df.style.apply(
        lambda x: ['background-color: yellow' if x.name == model_choice else '' for _ in x], axis=1)
    st.write("### Model Performance Metrics", results_df_styled)

    # Plotting with Plotly
    fig = go.Figure()

    # Residuals Plot
    for model_name in models.keys():
        y_pred = evaluation_results[model_name]['Predictions']
        residuals = y - y_pred
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name=f'Residuals - {model_name}'))

    fig.update_layout(title='Residuals Plot', xaxis_title='Predicted Values', yaxis_title='Residuals')
    st.plotly_chart(fig)

    # QQ Plot for Normality
    fig_qq = plt.figure(figsize=(12, 9))
    ax1 = fig_qq.add_subplot(211)

    stats.probplot(y - evaluation_results[model_choice]['Predictions'], dist="norm", plot=ax1)
    ax1.set_title('QQ Plot')

    st.pyplot(fig_qq)

    # Summary Explainer
    st.write("""
    ### Model Performance Summary
    The above metrics provide an overview of how well each model is performing:
    - **MAE (Mean Absolute Error)**: Measures the average magnitude of errors in the predictions, without considering their direction.
    - **MSE (Mean Squared Error)**: Similar to MAE but gives higher weight to larger errors.
    - **RMSE (Root Mean Squared Error)**: The square root of MSE, providing error in the same units as the target variable.
    - **R-Squared**: Indicates how well the model explains the variance in the target variable. A value closer to 1 is better.

    The residual plot shows how well the predicted values align with the actual values (ideally, residuals should be randomly scattered around zero).
    The QQ plot helps to assess if the residuals are normally distributed.
    """)

# Developer Credit
st.sidebar.write("Developed by Joseph Robinson")
