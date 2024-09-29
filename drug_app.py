#importing app library
import streamlit as st

#header
st.markdown(f"<h1 style='color: #90EE90;'>CANCER DRUG PREDICTION SYSTEM</h1>", unsafe_allow_html=True)

#sidebar
st.sidebar.title("DASHBOARD")
app_mode = st.sidebar.selectbox("Select The Appropriate Option",["Home Page","About Page","Important Feature Graph of Dataset","Prediction on User Inputs"])

#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('drug_dataset.csv')

# Create target variable based on the median of LN_IC50
data['TARGET'] = data['LN_IC50'].apply(lambda x: 1 if x < data['LN_IC50'].median() else 0)

# Prepare features and target
X = data[['GENE', 'PUTATIVE_TARGET', 'LN_IC50']]  
y = data['TARGET']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")  # Displays accuracy as a fraction

#home page
if(app_mode == "Home Page"):
    st.title("Home Page")
    image_path = "image.jpeg"
    img_path = "symbol.jpeg"
    st.image(image_path, use_column_width = True)
    st.markdown("### This App is Developed Because")
    st.markdown('''
1. As due the **Over Doses** of drugs the patient may get harashed due to their **Side Effects**.

2. We experienced at the time of covid that **Remdesivir Injection** is given to the people **Which is Not Required** due to which these people are now harashed due to **Bone and Many More Problems**.

3. Some people may get **Death** due to the **Over Doses or Unneccessary Doses** of drugs.

4. For **Avoiding this Responsiblity**, this app will give you **Prediction** about the patient of the given condition is **Sensitive or Resistive** for the drug and also give the name of the **Useful Drug** for the given tissue if the patient is **Sensitive** for the drug.

5. This app is **Useful for Doctors and Mediciens** as they have knowledge and experience regarding drugs and the conditions of the patient.                   
''')


#about page
elif(app_mode == "About Page"):
    st.title("About Page")
    st.markdown('''
1. Select the **Prediction on User Inputs** from the select box from sidebar for prediction purpose.
                
2. User has to give his inputs reagarding **Gene, Putative target, LN_IC50 and Tissue** for prediction purpose.
                
3. Click on **Prediction Button** which gives the output whether the drugs are **Sensitive or Resistive** for the given condition of patient.

4. If the output is **Sensitive** then it gives **Suggestions Regarding Which Drug is Useful** for the given tissue.

5. **Sensitivity and Resistivity Gragh** is also provided.

6. The **Graph of Important Features from the Dataset** is also provided and you can access it by selecting it from select box from sidebar
''')


#prediction page
elif(app_mode == "Prediction on User Inputs"):

    #accessing inputs
    st.title("Give Your Inputs")
    gene_input = st.text_input("Enter Gene Name (Capital Letters and Numeric Values)")
    putative_target_input = st.text_input("Enter Putative_target Name (Capital Letters and Numeric Values)")
    ln_ic50_input = st.number_input("Enter LN_IC50 (Floating Numeric Values)", format="%.10f", step=0.0001)
    tissue_input = st.text_input("Enter Tissue Name (Small Letters)")

    # Prepare the input data for prediction
    user_input = pd.DataFrame({
        'GENE': [gene_input],
        'PUTATIVE_TARGET': [putative_target_input],
        'LN_IC50': [ln_ic50_input]
    })

    # Convert categorical variables to dummy/indicator variables
    user_input = pd.get_dummies(user_input)
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    # Standardize the user input using the same scaler
    user_input_scaled = scaler.transform(user_input)

    # Predict the response using the trained model
    user_pred = rf_model.predict(user_input_scaled)

    # Display the result
    st.title("Prediction")
    if(st.button("Predict")):
        response = 'Sensitive' if user_pred[0] == 1 else 'Resistive'
        
        st.write(f'''Prediction for GENE: {gene_input},
        PUTATIVE_TARGET: {putative_target_input},
        LN_IC50: {ln_ic50_input}''')

        st.markdown(f"\n#### Predicted Response: ")
        if(response == "Sensitive"):
            st.markdown(f"<h1 style='color: red;'>{response}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: blue;'>{response}</h1>", unsafe_allow_html=True)

        # If the response is sensitive, suggest the best drug for the given tissue
        if response == 'Sensitive':
            sensitive_drugs = data[data['TARGET'] == 1]
            if sensitive_drugs[sensitive_drugs['TISSUE'] == tissue_input].empty:
                st.markdown(f"#### No effective drug found for tissue '{tissue_input}'.")
            else:
                common_drug = sensitive_drugs[sensitive_drugs['TISSUE'] == tissue_input]['DRUG_NAME'].value_counts().idxmax()
                st.markdown(f"#### The suggested effective drug for tissue '{tissue_input}' is: ")
                st.markdown(f"<h1 style='color: blue;'>{common_drug}</h1>", unsafe_allow_html=True)
        else:
            st.markdown("#### No specific drug suggestion as the response is Resistive.")


    #sensitivity and resistivity graph 
    st.title("Sensitivity and Resistivity Graph")
    if(st.button("Show Graph ")):
        ln_ic50_values = np.linspace(data['LN_IC50'].min(), data['LN_IC50'].max(), num=100)
        sensitivity = []
        resistivity = []

        for ln_ic50 in ln_ic50_values:
            # Prepare the input data for prediction
            user_input = pd.DataFrame({
                'GENE': [gene_input],
                'PUTATIVE_TARGET': [putative_target_input],
                'LN_IC50': [ln_ic50]
            })

            # Convert categorical variables to dummy/indicator variables
            user_input = pd.get_dummies(user_input)
            user_input = user_input.reindex(columns=X.columns, fill_value=0)

            # Standardize the user input using the same scaler
            user_input_scaled = scaler.transform(user_input)

            # Predict the response using the trained model
            user_pred = rf_model.predict(user_input_scaled)

            # Collect the responses
            if user_pred[0] == 1:
                sensitivity.append(1)
                resistivity.append(0)
            else:
                sensitivity.append(0)
                resistivity.append(1)

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the results
        ax.plot(ln_ic50_values, sensitivity, label='Sensitivity', color='green')
        ax.plot(ln_ic50_values, resistivity, label='Resistivity', color='red')
        ax.set_title(f'Sensitivity and Resistivity for {gene_input} and {putative_target_input}')
        ax.set_xlabel('LN_IC50 Values')
        ax.set_ylabel('Response')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Resistive', 'Sensitive'])
        ax.axhline(0.5, color='grey', linestyle='--', label='Threshold')
        ax.legend()
        ax.grid()
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

#feature importance graph
elif(app_mode == "Important Feature Graph of Dataset"):
    st.title("Important Feature Graph of Dataset")
    if(st.button("Show Graph")):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get feature importances and features
        feature_importances = rf_model.feature_importances_
        features = X.columns

        # Create a DataFrame for feature importance
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plotting
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance for Drug Response Prediction')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        ## ax is taken instead of plt for plotting the graph because we are using streamlit library for developing an app