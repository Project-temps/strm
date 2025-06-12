# stakeholder_page.py

import streamlit as st

def stakeholder_page():
    st.title("Stakeholder Page")
    st.write("Welcome, Stakeholder!")

    # Logout button
    if st.button('Logout'):
        if 'user' in st.session_state:
            del st.session_state.user  # Remove the user from session state
        st.experimental_rerun()  # Rerun the app to navigate back to the public page

    # Your stakeholder page logic here
    # ...
