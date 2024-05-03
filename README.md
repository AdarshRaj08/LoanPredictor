# Loan Prediction Web Application

This project is a web application built to assist lenders in quickly assessing loan applications. It predicts loan approval based on various applicant features such as income, credit history, and property location. The application streamlines the lending process, enabling faster decisions.

## Features

- Exploratory Data Analysis (EDA): The application includes an EDA section where data insights are visualized through various plots, providing a deeper understanding of the dataset.
- Machine Learning Model: Utilizes a Random Forest Classifier model trained on loan application data to predict loan approval.
- Streamlit Deployment: The web application is deployed using Streamlit, allowing easy access through a web browser.

## Dataset Information

The dataset used in this project was sourced from Kaggle and contains information about loan applicants, including their demographics, financial history, and loan status. The model achieved an accuracy of 85% on the test dataset.

## Usage

To run the application locally:

1. Clone this repository to your local machine.
2. Ensure you have Python installed.
3. Install the required dependencies listed in `requirements.txt`.
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app using the following command:
   ```
   streamlit run app.py
   ```

## File Structure

- `app.py`: Contains the Streamlit application code, including the EDA and prediction functionalities.
- `loan.pkl`: Pickled machine learning model trained on loan application data.
- `dataset.csv`: Dataset containing loan application information.

## Dependencies

- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Credits

- Dataset Source: [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication/data)
- Model Training: Random Forest Classifier from Scikit-learn
