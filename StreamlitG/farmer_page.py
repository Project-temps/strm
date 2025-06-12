# # farmer_page.py

# import streamlit as st

# def farmer_page():
#     st.title("Farmer Page")
#     st.write("Welcome, Farmer!")

#     # Logout button
#     if st.button('Logout'):
#         if 'user' in st.session_state:
#             del st.session_state.user  # Remove the user from session state
#         st.experimental_rerun()  # Rerun the app to navigate back to the public page

#     # Your farmer page logic here
#     # ...

import streamlit as st

def farmer_page():
    # Use more specific CSS selectors to ensure the background color is applied
    st.markdown("""
        <style>
        /* Target the main content area */
        .reportview-container .main .block-container{
            background-color: #E8F5E9; /* Light green background */
        }
        /* Target the overall page background, including padding areas */
        body {
            background-color: #C8E6C9; /* Slightly darker green background for the whole page */
        }
        /* Update text and button colors */
        .reportview-container {
            color: #388E3C; /* Darker green text */
        }
        .stButton>button {
            background-color: #4CAF50; /* Green button */
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)


    st.title("Farmer Dashboard")
    st.write("Welcome to the Farmer Dashboard. This page is themed with green colors to reflect the farming context.")

    # Logout button
    if st.button('Logout'):
        if 'user' in st.session_state:
            del st.session_state.user  # Remove the user from session state
        st.experimental_rerun()  # Rerun the app to navigate back to the public page
   
    # Your code for the farmer page goes here

