import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time # Added for spinner demo

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- Custom CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# We will create a style.css file, but for a single file, let's inject it directly
st.markdown("""
<style>
/* Main container for the app */
.stApp {
    background-color: black; /* Light gray background */
}

/* Create a card-like container for the prediction form */
div.st-emotion-cache-1r6slb0 { /* This selector targets the main block container */
    background-color: #ffffff;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border: 1px solid #e6e6e6;
}

/* Style the title */
h1 {
    color: #1a1a1a;
    text-align: center;
}

/* Style the button */
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    background-color: #0068c9; /* A nice blue */
    color: white;
    border: none;
    padding: 10px 0;
    font-weight: 600;
}
div.stButton > button:hover {
    background-color: #005cb3;
    color: white;
    border: none;
}

/* Style success and error messages */
div.stAlert.st-emotion-cache-l00688.e1k6c8200, /* Success box */
div.stAlert.st-emotion-cache-l00688.e8s6d3d0 {  /* Error box */
    border-radius: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
try:
    # Load the pre-trained ML model
    model_container = joblib.load('Decision_Tree_model_joblib.pkl')
    # Load the scaler
    scaler = joblib.load('scaler.pkl')
    # Load the columns used in the model
    columns = joblib.load('columns.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- Model Resolver ---
actual_model = None
if isinstance(model_container, dict):
    if 'model' in model_container:
        actual_model = model_container['model']
    else:
        for key, value in model_container.items():
            if hasattr(value, 'predict'):
                actual_model = value
                st.info(f"Note: Using model object found under key '{key}' in .pkl file.")
                break
    
    if actual_model is None:
        st.error("Error: Loaded model file is a dictionary, but no valid model object with a '.predict()' method was found inside it.")
        st.stop()
elif hasattr(model_container, 'predict'):
    actual_model = model_container
else:
    st.error("Error: Loaded model file is not a dictionary and does not appear to be a valid model object.")
    st.stop()
# --- End Model Resolver ---


# --- App UI ---
st.title('🎓 Placement Predictor')
st.write('Predict whether a student will be placed based on their academic and personal attributes.')

# --- Prediction Form ---
# Using st.form stops the app from re-running on every widget change
with st.form(key='prediction_form'):
    st.subheader("Enter Student Details")

    # Use columns for a cleaner layout (2 inputs per row)
    col1, col2 = st.columns(2)
    
    with col1:
        IQ = st.number_input('🧠 Enter IQ Score', min_value=0, max_value=200, value=100)
        Academic_Performance = st.number_input('📚 Academic Performance (1-10)', min_value=1, max_value=10, value=5)
        PROJECTS_COMPLETED = st.number_input('💻 Projects Completed', min_value=0, max_value=50, value=0)
    
    with col2:
        CGPA = st.number_input('📊 Enter CGPA', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        Intenrnship_Experience = st.selectbox('👔 Internship Experience', options=['No', 'Yes'])
        COMUNICATION_SKILLS = st.number_input('🗣️ Communication Skills (1-10)', min_value=1, max_value=10, value=5)

    # The submit button for the form
    submit_button = st.form_submit_button(label='Predict Placement')

# --- Prediction Logic (runs only when form is submitted) ---
if submit_button:
    with st.spinner('Analyzing profile...'): # Loading animation
        time.sleep(1) # Small delay to make spinner visible
        
        try:
            # Prepare the input data
            input_data = pd.DataFrame(columns=columns)
            input_data.loc[0] = np.zeros(len(columns))
            
            input_data.at[0, 'IQ'] = IQ
            input_data.at[0, 'CGPA'] = CGPA
            input_data.at[0, 'Academic_Performance'] = Academic_Performance
            input_data.at[0, 'Projects_Completed'] = PROJECTS_COMPLETED
            input_data.at[0, 'Communication_Skills'] = COMUNICATION_SKILLS
            
            if Intenrnship_Experience == 'Yes':
                input_data.at[0, 'Internship_Experience'] = 1
            else:
                input_data.at[0, 'Internship_Experience'] = 0

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make prediction using the resolved model
            prediction = actual_model.predict(input_data_scaled)

            # Display the result
            if prediction[0] == 1:
                st.success('🎉 Likelihood: **Placed**')
                st.balloons() # Celebration animation
                                
            else:
                st.error('📉 Likelihood: **Not Placed**')
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            # st.exception(e) # Uncomment for detailed traceback in the app

# --- About Section ---

initial_sidebar_state="expanded" 

with st.sidebar: 
    st.header("About")
    st.write('This app uses a **Decision Tree Classifier** to predict student placements based on various attributes.\n ' \
    'the model predicts whether a student is likely to be placed or not based on inputs like IQ, CGPA, academic performance, projects completed, internship experience, and communication skills **this is mot a real placement predictor and is for educational purposes only.**  ')
    st.info('Developed by Mayank Singh Parihar.')

