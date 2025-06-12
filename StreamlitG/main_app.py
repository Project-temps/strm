# main_app.py

import streamlit as st
from public_page import public_page
from farmer_page import farmer_page
from stakeholder_page import stakeholder_page

def main():
    # Get the current session state
    session_state = st.session_state

    # Check if the user is logged in and determine their role
    if 'user' not in session_state:
        session_state.user = {'role': 'public'}  # Default role is public

    # Render different pages based on user role
    if session_state.user['role'] == 'public':
        public_page()
    elif session_state.user['role'] == 'farmer':
        farmer_page()
    elif session_state.user['role'] == 'stakeholder':
        stakeholder_page()

# Run the main app
if __name__ == "__main__":
    main()


# streamlit run main_app.py


