import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open('loan.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


df = pd.read_csv('dataset.csv')
st.markdown("""
    <div style='background-color:#f9f9f9;padding:10px;border-radius:10px'>
    <h2 style='text-align:center;color:#dc143c;'>Welcome to LoanPredict</h2>
    <p style='text-align:justify;color:#5f6368;'>LoanPredict is a web app I built to help lenders quickly assess loan applications. It predicts loan approval based on applicant data like income, credit history, and property location. It streamlines the lending process for faster decisions.</p>
    </div>
    """, unsafe_allow_html=True)

# Function for the EDA page
def eda_page():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Exploratory Data Analysis')
    

    col1, col2 = st.columns(2)

    # Add a plot to the first column
    with col1:
        st.subheader('Relation b/w Education and loan status')
        fig, ax = plt.subplots()
        sns.countplot(x='Education',hue='Loan_Status',data=df)
        st.pyplot(fig)

    # Add some text to the second column
    with col2:
        st.subheader('Relation b/w Self Employed and loan status')
        fig, ax = plt.subplots()
        sns.countplot(x='Married',hue='Loan_Status',data=df)
        st.pyplot(fig)

    # second row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Relation b/w Self Employed and loan status')
        fig, ax = plt.subplots()
        sns.countplot(x='Self_Employed',hue='Loan_Status',data=df)
        st.pyplot(fig)

    with col2:
        st.subheader('Relation b/w Property Area and loan status')
        fig, ax = plt.subplots()
        sns.countplot(x='Property_Area',hue='Loan_Status',data=df)
        st.pyplot(fig)

    # third row
    st.subheader('Distribution of Loan amount Term')
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.hist(df['Loan_Amount_Term'], bins=15, color='red', edgecolor='black')
    plt.xlabel('Loan_Amount_Term(in months)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Loan Amount Term')
    st.pyplot(fig)

    # fourth row
    st.subheader('Distribution of Loan amount')
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.hist(df['LoanAmount'], bins=15, color='red', edgecolor='black')
    plt.xlabel('Loan_Amount(in $)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Loan Amount')
    st.pyplot(fig)

    # fifth row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Relation b/w Gender and loan status')
        fig, ax = plt.subplots()
        sns.countplot(x='Gender',hue='Loan_Status',data=df)
        st.pyplot(fig)

    with col2:
        st.subheader('Relation b/w Credit History and loan status')
        fig, ax = plt.subplots()
        sns.countplot(x='Credit_History',hue='Loan_Status',data=df)
        st.pyplot(fig)


    # sixth row
    st.subheader('Distribution of Applicant Income($)')
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.hist(df['ApplicantIncome'],bins=10, color='red', edgecolor='black')
    plt.xlabel('ApplicantIncome($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of ApplicantIncome')
    st.pyplot(fig)

    # seventh row
    st.subheader('Relation of Dependents with Loan status')
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x='Dependents',hue='Loan_Status',data=df)

    plt.xlabel('ApplicantIncome($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of ApplicantIncome')
    st.pyplot(fig)





# Function for the model prediction page
def prediction_page():
    st.title('Loan Prediction Model')

    st.header('Enter Details for Loan Prediction')

    # Create separate form fields for each feature
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income')
    coapplicant_income = st.number_input('Coapplicant Income')
    loan_amount = st.number_input('Loan Amount')
    loan_amount_term = st.number_input('Loan Amount Term')
    credit_history = st.selectbox('Credit History', ['1', '0'])
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

    # Convert categorical values to numerical for prediction
    if st.button('Predict'):
        gender = 1 if gender == 'Male' else 0
        married = 1 if married == 'Yes' else 0
        education = 1 if education == 'Graduate' else 0
        self_employed = 1 if self_employed == 'Yes' else 0
        property_area = 2 if property_area == 'Urban' else (1 if property_area == 'Semiurban' else 0)
        
        # Make prediction
        prediction = model.predict([[gender, married, dependents, education, self_employed, applicant_income,
                                    coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])
        
        # Display prediction result
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.write('Loan Approved')
        else:
            st.write('Loan Denied')

# Main function to run the Streamlit app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select a page', ['EDA', 'Prediction'])

    if page == 'EDA':
        eda_page()
    elif page == 'Prediction':
        prediction_page()

if __name__ == '__main__':
    main()
