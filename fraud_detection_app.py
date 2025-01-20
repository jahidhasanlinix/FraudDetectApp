import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                           accuracy_score, precision_score, recall_score, f1_score)
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
# Tested it again
st.set_page_config(page_title="Credit Card Fraud Detection System", layout="wide")

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the uploaded dataset."""
    df = pd.read_csv(uploaded_file)
    df = df.drop_duplicates()
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    if len(numeric_columns) > 0:
        numeric_imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    return df

def prepare_features(df, target_column):
    """Prepare features for model training."""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert categorical variables to dummy variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns)
  # testing
    
    return X, y

def train_model(X, y):
    """Train the Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

def create_visualizations(predictions, y_test, y_pred_proba, feature_names, model):
    """Create comprehensive visualizations for model evaluation"""
    
    # 1. Confusion Matrix with Plotly
    cm = confusion_matrix(y_test, predictions)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Not Fraud', 'Fraud'],
        y=['Not Fraud', 'Fraud'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        colorscale='Blues'
    ))
    
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=500,
        height=500
    )
    
    # 2. ROC Curve with Plotly
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure(data=go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        mode='lines'
    ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Random'
    ))
    
    fig_roc.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=500,
        height=500,
        showlegend=True
    )
    
    # 3. Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    fig_importance = px.bar(
        importance_df,
        x='importance',
        y='feature',
        title="Top 10 Important Features",
        orientation='h'
    )
    
    # 4. Score Distribution with Plotly
    fraud_probs = y_pred_proba[y_test == 1]
    non_fraud_probs = y_pred_proba[y_test == 0]
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=non_fraud_probs,
        name='Not Fraud',
        opacity=0.75,
        nbinsx=50,
        histnorm='probability density'
    ))
    fig_dist.add_trace(go.Histogram(
        x=fraud_probs,
        name='Fraud',
        opacity=0.75,
        nbinsx=50,
        histnorm='probability density'
    ))
    
    fig_dist.update_layout(
        title="Distribution of Predicted Probabilities",
        xaxis_title="Predicted Probability of Fraud",
        yaxis_title="Density",
        barmode='overlay',
        width=500,
        height=500
    )
    
    return fig_cm, fig_roc, fig_importance, fig_dist

def display_metrics(y_test, predictions, y_pred_proba):
    """Display comprehensive model performance metrics"""
    metrics_dict = {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'F1 Score': f1_score(y_test, predictions),
        'ROC AUC': auc(*roc_curve(y_test, y_pred_proba)[:2])
    }
    
    metrics_df = pd.DataFrame({
        'Metric': metrics_dict.keys(),
        'Value': [f"{v:.3f}" for v in metrics_dict.values()]
    })
    
    fraud_summary = {
        'Total Transactions': len(y_test),
        'Actual Fraud Cases': np.sum(y_test == 1),
        'Detected Fraud Cases': np.sum((y_test == 1) & (predictions == 1)),
        'False Alarms': np.sum((y_test == 0) & (predictions == 1)),
        'Fraud Detection Rate': f"{(np.sum((y_test == 1) & (predictions == 1))/np.sum(y_test == 1))*100:.1f}%"
    }
    
    summary_df = pd.DataFrame({
        'Metric': fraud_summary.keys(),
        'Value': fraud_summary.values()
    })
    
    return metrics_df, summary_df


# Main app
def main():
    st.title("Credit Card Fraud Detection System")
    st.write("""
    This application detects potential credit card fraud using machine learning.
    Upload your dataset or use the default Credit Card Fraud dataset.
    """)

    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

    if uploaded_file is not None:
        # Load data
        df = load_and_preprocess_data(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())
        
        # Select target column
        target_column = st.selectbox(
            "Select the column that indicates fraud (1 for fraud, 0 for non-fraud)",
            df.columns
        )
        
        # Add model parameters selection
        st.sidebar.header("Model Parameters")
        n_estimators = st.sidebar.slider("Number of trees", 50, 200, 100)
        max_depth = st.sidebar.slider("Maximum depth", 10, 100, 30)
        min_samples_split = st.sidebar.slider("Minimum samples split", 2, 10, 2)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Prepare features and target
                X, y = prepare_features(df, target_column)
                
                # Train model with selected parameters
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    class_weight='balanced'
                )
                
                # Split and scale data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Fit model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                predictions = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Display classification report
                st.write("### Model Performance")
                report = classification_report(y_test, predictions, zero_division=0)
                st.code(report)
                
                # Display metrics
                metrics_df, summary_df = display_metrics(y_test, predictions, y_pred_proba)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Performance Metrics")
                    st.table(metrics_df)
                with col2:
                    st.write("### Fraud Detection Summary")
                    st.table(summary_df)
                
                # Create and display visualizations
                fig_cm, fig_roc, fig_importance, fig_dist = create_visualizations(
                    predictions, y_test, y_pred_proba, X.columns, model)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.plotly_chart(fig_cm)
                    st.plotly_chart(fig_roc)
                with col4:
                    st.plotly_chart(fig_importance)
                    st.plotly_chart(fig_dist)
                
                # Add download button for predictions
                predictions_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': predictions,
                    'Probability': y_pred_proba
                })
                st.download_button(
                    label="Download Predictions",
                    data=predictions_df.to_csv(index=False),
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

    st.markdown("""
    ### How to Use This App:
    1. Upload a CSV file containing credit card transaction data
    2. Select the column that indicates fraud (should be binary: 0 for non-fraud, 1 for fraud)
    3. Adjust model parameters in the sidebar (optional)
    4. Click 'Train Model' to start the fraud detection process
    5. Review the results and visualizations
    6. Download the predictions for further analysis
    """)

# Run the app
if __name__ == '__main__':
    main()
