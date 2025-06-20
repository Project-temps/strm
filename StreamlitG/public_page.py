# public_page.py
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from user_credentials import FARMERS, STAKEHOLDERS


def login_section():
    """Displays the login section in the sidebar with expanders for different user roles."""
    with st.sidebar:
        st.subheader("Login")
        login_type = st.radio("Login as", ('Farmer', 'Stakeholder'))

        if login_type == 'Farmer':
            farmer_username = st.text_input("Username", key='farmer_username', placeholder="Enter username")
            farmer_password = st.text_input("Password", type='password', key='farmer_password', placeholder="Enter password")

            if st.button("Login", key='farmer_login_button'):
                js = """
                    <script>
                    window.history.pushState({}, "", "/farmer");
                    </script>
                    """
                st.markdown(js, unsafe_allow_html=True)
                if farmer_username in FARMERS and FARMERS[farmer_username] == farmer_password:
                    st.session_state.user = {'role': 'farmer'}  # Successful login, set user role to farmer
                    st.success("Logged in successfully as a farmer.")
                    st.experimental_rerun()  # Rerun the app to navigate to the farmer page
                else:
                    st.error("Incorrect username or password for farmer.")

        elif login_type == 'Stakeholder':
            stakeholder_username = st.text_input("Username", key='stakeholder_username', placeholder="Enter username")
            stakeholder_password = st.text_input("Password", type='password', key='stakeholder_password', placeholder="Enter password")

            if st.button("Login", key='stakeholder_login_button'):
                if stakeholder_username in STAKEHOLDERS and STAKEHOLDERS[stakeholder_username] == stakeholder_password:
                    st.session_state.user = {'role': 'stakeholder'}  # Successful login, set user role to stakeholder
                    st.success("Logged in successfully as a stakeholder.")
                    st.experimental_rerun()  # Rerun the app to navigate to the stakeholder page
                else:
                    st.error("Incorrect username or password for stakeholder.")


def public_content():
    """Displays the content of the public page."""
    st.title("Welcome to the Prediction Platform")
    st.markdown("""
        This platform provides predictive insights for farmers and stakeholders.
        Please log in using the sidebar to access your personalized dashboard.
        """)
    # Load the trained model
    model = load_model('model.h5')

    # Load the scaler model
    scalerfile = 'scaler.pkl'
    # scaler = pickle.load(open(scalerfile, 'rb'))
    scaler = joblib.load("scaler.pkl")
    target_scaler = joblib.load("target_scaler.pkl")

    # Streamlit UI
    st.title("Prediction App for farm time series data")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file for predictions", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col=[0])

        # Ensure the input features match those used during training
        input_columns = [
            "CH4_w-out", "CH4_s-out", "CH4_n-out", "CH4_e-out", "CH4_n-in",
            "CH4_m-in", "CH4_m-in-up", "CH4_s-in", "CH4_w-in", "CH4_e-in",
            "TEMP", "Ver_w", "Hor_w", "CH4_in_mean", "CH4_out_mean",
            "CO2_n-in", "CO2_m-in", "CO2_m-in-up", "CO2_s-in",
            "CO2_w-in", "CO2_e-in"
        ]
        df = df[input_columns]

        test_split = round(len(df) * 0.20)

        # Create training and testing datasets
        df_for_training = df[:-test_split]
        df_for_testing = df[-test_split:]

        # Initialize MinMaxScaler and scale the training and testing datasets
        df_for_training_scaled = scaler.transform(df_for_training)
        df_for_testing_scaled = scaler.transform(df_for_testing)

        # Define a function to create input features (dataX) and target variable (dataY) for time series prediction
        def createXY(dataset, n_past, n_future):
            dataX = []
            for i in range(len(dataset) - n_past - n_future + 1):
                dataX.append(dataset[i:(i + n_past), :])
            return np.array(dataX)

        # Set the number of past and future time steps to match the training configuration
        n_past = 168
        n_future = 12  # Predict future hours

        # Generate testing data
        testX = createXY(df_for_testing_scaled, n_past, n_future)

        inputX = testX

        # Generate predictions
        predictions = model.predict(inputX)
        num_sequences = predictions.shape[0]
        predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))
        predictions_flat = predictions.reshape(-1)

        # Compute actual CH4 values aligned with predictions
        actual_ch4, dates = [], []
        target = df_for_testing["CH4_in_mean"].values
        for i in range(len(target) - n_past - n_future + 1):
            for j in range(n_future):
                actual_ch4.append(target[i + n_past + j])
                dates.append(df_for_testing.index[i + n_past + j])

        predictions_df = pd.DataFrame({
            "Date": dates,
            "Actual_CH4_in_mean": actual_ch4,
            "Predicted_CH4_in_mean": predictions_flat,
        })

        st.write("### Predicted Results:")
        st.dataframe(predictions_df)

        # Allow the user to choose which columns to display for input data visualization
        st.write("### Input Data Visualization for manual adjustment:")
        selected_input_columns = st.multiselect("Select input columns to display", df.columns)
        
        # Plotly figure for interactive input adjustment
        fig_input = px.line(df_for_testing, x=df_for_testing.index, y=selected_input_columns, title="Input Data Visualization")
        st.plotly_chart(fig_input, use_container_width=True)

        # Display prediction plot for CH4_in_mean
        st.write("### Predictions:")
        plt.figure(figsize=(10, 5))
        plt.plot(predictions_df["Date"], predictions_df["Actual_CH4_in_mean"], label='Actual CH4_in_mean', color='red')
        plt.plot(predictions_df["Date"], predictions_df["Predicted_CH4_in_mean"], label='Predicted CH4_in_mean', color='blue')
        plt.title('Actual vs Predicted CH4_in_mean')
        plt.xlabel('Date')
        plt.ylabel('CH4_in_mean')
        plt.legend()
        st.pyplot(plt)


def public_page():
    """Main function for the public page."""
    st.set_page_config(page_title="Prediction Project - ET4D", page_icon="🌐", layout="wide")
    login_section()
    public_content()

if __name__ == "__main__":
    public_page()


