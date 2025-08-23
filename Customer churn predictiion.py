import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Data Overview"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .prediction-low {
        color: #006400;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sample data for demonstration
@st.cache_data
def load_sample_data():
    data = {
        'Age': [45, 38, 47, 58, 37, 29, 52, 41, 35, 44, 33, 55, 26, 60, 48],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 
                  'Male', 'Female', 'Male', 'Female', 'Male'],
        'Tenure': [43, 1, 45, 51, 24, 3, 60, 12, 6, 36, 18, 48, 9, 54, 30],
        'Usage Frequency': [20, 18, 18, 2, 4, 15, 22, 8, 12, 19, 10, 25, 7, 5, 16],
        'Support Calls': [4, 10, 7, 2, 4, 5, 1, 9, 3, 2, 6, 1, 8, 3, 2],
        'Payment Delay': [24, 12, 26, 5, 30, 15, 0, 22, 18, 7, 25, 3, 28, 10, 14],
        'Subscription Type': ['Standard', 'Standard', 'Basic', 'Standard', 'Standard', 
                             'Premium', 'Premium', 'Basic', 'Standard', 'Premium',
                             'Basic', 'Premium', 'Standard', 'Basic', 'Premium'],
        'Contract Length': ['Annual', 'Annual', 'Quarterly', 'Annual', 'Quarterly',
                           'Monthly', 'Annual', 'Quarterly', 'Monthly', 'Annual',
                           'Quarterly', 'Annual', 'Monthly', 'Annual', 'Quarterly'],
        'Total Spend': [127.32, 859.43, 514.30, 256.83, 220.40, 450.25, 999.99, 312.75, 185.60, 789.45,
                       345.67, 876.54, 234.56, 765.43, 543.21],
        'Churn': [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    }
    return pd.DataFrame(data)

# Train a model for demonstration
@st.cache_resource
def train_model(_X_train, _y_train, model_type='Random Forest'):
    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'SVM':
        model = SVC(random_state=42, probability=True)
    
    model.fit(_X_train, _y_train)
    return model

# Preprocess data function
def preprocess_data(df, is_training=False):
    df = df.copy()
    le = LabelEncoder()
    
    # Encode categorical variables
    if 'Gender' in df.columns:
        df['Gender'] = le.fit_transform(df['Gender'])
    if 'Subscription Type' in df.columns:
        df['Subscription Type'] = le.fit_transform(df['Subscription Type'])
    if 'Contract Length' in df.columns:
        df['Contract Length'] = le.fit_transform(df['Contract Length'])
    
    return df

# Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select a page:", [
    "Data Overview", 
    "Exploratory Data Analysis", 
    "Data Preprocessing", 
    "Model Training", 
    "Predictions",
    "Predict My Churn"
])

# Main title
st.markdown('<h1 class="main-header">Customer Churn Prediction Analysis</h1>', unsafe_allow_html=True)
st.write("This application analyzes customer data to predict churn behavior. Upload your dataset or use the sample data to explore the analysis.")

# Load sample data
df = load_sample_data()

if page == "Data Overview":
    st.header("📋 Data Overview")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File successfully uploaded!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.info("Using sample data instead.")
            df = load_sample_data()
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Number of Features", len(df.columns))
        st.metric("Churn Rate", f"{(df['Churn'].sum() / len(df)) * 100:.2f}%")
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes.astype(str))
        
        st.write("**Missing Values:**")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")

elif page == "Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(df)
    
    # Visualizations
    st.subheader("Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)
        ax.set_title('Churn Distribution')
        ax.set_xlabel('Churn')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with col2:
        # Age distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Age'].hist(bins=15, color='lightgreen', ax=ax)
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    # Feature distributions by churn
    st.subheader("Feature Distributions by Churn Status")
    feature = st.selectbox("Select feature to analyze", ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Churn', y=feature, data=df, ax=ax)
    ax.set_title(f'{feature} Distribution by Churn Status')
    st.pyplot(fig)

elif page == "Data Preprocessing":
    st.header("⚙️ Data Preprocessing")
    
    st.subheader("Current Data")
    st.dataframe(df.head())
    
    st.subheader("Data Preprocessing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Handle missing values"):
            # Simple imputation for demonstration
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            
            st.success("Missing values handled!")
    
    with col2:
        if st.checkbox("Encode categorical variables"):
            # This is done in the preprocess function
            st.info("Categorical variables will be encoded during model training")
    
    if st.checkbox("Show preprocessed data"):
        processed_df = preprocess_data(df)
        st.subheader("Preprocessed Data")
        st.dataframe(processed_df.head())
        
        st.download_button(
            label="Download Preprocessed Data",
            data=processed_df.to_csv(index=False),
            file_name="preprocessed_churn_data.csv",
            mime="text/csv"
        )

elif page == "Model Training":
    st.header("🤖 Model Training")
    
    # Preprocess data
    processed_df = preprocess_data(df)
    X = processed_df.drop('Churn', axis=1)
    y = processed_df['Churn']
    
    # Train-test split
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Model selection
    model_type = st.selectbox("Select model", ["Random Forest", "Logistic Regression", "SVM"])
    
    if st.button("Train Model"):
        with st.spinner(f"Training {model_type} model..."):
            model = train_model(X_train, y_train, model_type)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success("Model trained successfully!")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy:.2%}")
            col2.metric("Precision", f"{accuracy_score(y_test, y_pred):.2%}")
            col3.metric("Recall", f"{accuracy_score(y_test, y_pred):.2%}")
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

elif page == "Predictions":
    st.header("🔮 Batch Predictions")
    
    st.info("Upload a CSV file with customer data to get churn predictions for multiple customers.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_predict")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_df = pd.read_csv(uploaded_file)
            st.success("✅ File successfully uploaded!")
            
            # Display the uploaded data
            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head())
            
            # Check if required columns are present
            required_columns = ['Age', 'Gender', 'Tenure', 'Usage Frequency', 
                               'Support Calls', 'Payment Delay', 'Subscription Type', 
                               'Contract Length', 'Total Spend']
            
            if all(col in batch_df.columns for col in required_columns):
                st.success("All required columns are present.")
                
                # Preprocess the data
                processed_batch = preprocess_data(batch_df)
                
                # Preprocess sample data to get the model
                processed_sample = preprocess_data(df)
                X_sample = processed_sample.drop('Churn', axis=1)
                y_sample = processed_sample['Churn']
                
                # Train a model
                model = train_model(X_sample, y_sample)
                
                # Make predictions
                predictions = model.predict(processed_batch)
                prediction_proba = model.predict_proba(processed_batch)
                
                # Add predictions to the dataframe
                batch_df['Churn Prediction'] = predictions
                batch_df['Churn Probability'] = prediction_proba[:, 1]
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(batch_df)
                
                # Summary statistics
                churn_count = batch_df['Churn Prediction'].sum()
                churn_rate = churn_count / len(batch_df)
                
                st.subheader("Prediction Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Customers", len(batch_df))
                col2.metric("Predicted Churns", churn_count)
                col3.metric("Churn Rate", f"{churn_rate:.2%}")
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
            else:
                missing_cols = [col for col in required_columns if col not in batch_df.columns]
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns are: Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    else:
        st.info("Please upload a CSV file to get predictions.")
        
        # Show sample data format
        st.subheader("Expected Data Format")
        sample_df = load_sample_data().drop('Churn', axis=1)
        st.dataframe(sample_df.head())
        
        # Provide sample data download
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=sample_csv,
            file_name="sample_churn_data.csv",
            mime="text/csv"
        )

elif page == "Predict My Churn":
    st.header("👤 Predict My Churn")
    st.write("Enter the customer information below to predict the likelihood of churn.")
    
    # Create a form for user input
    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=18, max_value=100, value=45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=24)
            usage_frequency = st.slider("Usage Frequency (times/month)", min_value=0, max_value=30, value=15)
            support_calls = st.slider("Support Calls (last month)", min_value=0, max_value=20, value=3)
            
        with col2:
            payment_delay = st.slider("Payment Delay (days)", min_value=0, max_value=60, value=15)
            subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            total_spend = st.slider("Total Spend ($)", min_value=0, max_value=2000, value=500)
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Tenure': [tenure],
            'Usage Frequency': [usage_frequency],
            'Support Calls': [support_calls],
            'Payment Delay': [payment_delay],
            'Subscription Type': [subscription_type],
            'Contract Length': [contract_length],
            'Total Spend': [total_spend]
        })
        
        # Display the user input
        st.subheader("Your Input")
        st.dataframe(user_data)
        
        # Preprocess the data
        processed_data = preprocess_data(user_data)
        
        # Preprocess sample data to get the model
        processed_sample = preprocess_data(df)
        X_sample = processed_sample.drop('Churn', axis=1)
        y_sample = processed_sample['Churn']
        
        # Train a model
        model = train_model(X_sample, y_sample)
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Display results
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.error("⚠️ High risk of churn! This customer is likely to leave.")
            st.write(f"Probability of churn: {prediction_proba[0][1]:.2%}")
            
            # Show reasons for churn risk
            st.write("**Potential reasons for churn risk:**")
            reasons = []
            if payment_delay > 30:
                reasons.append("Payment delay is too high")
            if support_calls > 10:
                reasons.append("Many support calls indicate dissatisfaction")
            if usage_frequency < 5:
                reasons.append("Low usage frequency")
            if tenure < 6:
                reasons.append("Short tenure (new customer)")
                
            for reason in reasons:
                st.write(f"- {reason}")
                
            st.write("**Recommendations:**")
            st.write("- Offer a discount or promotion")
            st.write("- Provide personalized support")
            st.write("- Check if features meet customer needs")
            
        else:
            st.success("✅ Low risk of churn. This customer is likely to stay.")
            st.write(f"Probability of churn: {prediction_proba[0][1]:.2%}")
            
            st.write("**This customer is satisfied because:**")
            positives = []
            if payment_delay <= 7:
                positives.append("Prompt payments")
            if support_calls <= 2:
                positives.append("Few support requests")
            if usage_frequency >= 15:
                positives.append("High product usage")
            if tenure >= 12:
                positives.append("Long-term relationship")
                
            for positive in positives:
                st.write(f"- {positive}")