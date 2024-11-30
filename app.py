import streamlit as st
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import pickle

# Load the trained model (make sure the file path is correct)
filename = 'purchase_behavior_model.pkl'  # Replace with your file path
with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)

# Load the clustered customer data (make sure the file path is correct)
df = pd.read_csv("customer_purchase_with_clusters.csv")

# Set up the Streamlit page appearance
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('<style>body{background-color: #ADD8E6;}</style>', unsafe_allow_html=True)
st.title("Customer Purchase Behavior Prediction")

# Input form for customer features (match features with your project)
with st.form("customer_form"):
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    gender = st.radio("Gender", options=["Male", "Female"], index=0)
    annual_income = st.number_input('Annual Income ($)', min_value=0, value=50000, step=1000)
    num_purchases = st.number_input('Number of Purchases', min_value=0, value=10)
    product_category = st.selectbox('Product Category', ['Electronics', 'Clothing', 'Home Goods', 'Beauty', 'Sports'])
    time_spent_on_website = st.number_input('Time Spent on Website (minutes)', min_value=0, value=20)
    loyalty_program = st.selectbox('Loyalty Program Member', ['No', 'Yes'])
    discounts_availed = st.number_input('Discounts Availed', min_value=0, max_value=5, value=2)

    # Collect the input data
    data = [[age, 1 if gender == "Female" else 0, annual_income, num_purchases,
             ['Electronics', 'Clothing', 'Home Goods', 'Beauty', 'Sports'].index(product_category),
             time_spent_on_website, 1 if loyalty_program == 'Yes' else 0, discounts_availed]]

    # Form submit button
    submitted = st.form_submit_button("Submit")

# After submission, process the data and display cluster information
if submitted:
    # Predict the cluster using the trained model
    predicted_cluster = loaded_model.predict(data)[0]
    
    # Display the predicted cluster number
    st.write(f'Customer belongs to Cluster: {predicted_cluster}')
    
    # Map cluster names and descriptions (adjust the descriptions as needed)
    cluster_mapping = {
        0: ("Frequent Big Spenders", "These customers have a high likelihood of making purchases, spend a lot, and engage with promotions."),
        1: ("Occasional Shoppers", "These customers make occasional purchases, generally responding well to specific offers."),
        2: ("Discount Seekers", "These customers are more likely to make a purchase if discounts are available."),
        3: ("Low Engagement", "These customers show low engagement with the website and have a low likelihood of making a purchase.")
    }

    cluster_name, cluster_description = cluster_mapping.get(predicted_cluster, ("Unknown Cluster", "No description available"))
    
    # Display cluster name and description
    st.write(f"**Cluster Name**: {cluster_name}")
    st.write(f"**Description**: {cluster_description}")
    
    # Filter the DataFrame for the selected cluster
    cluster_df = df[df['Cluster'] == predicted_cluster]

    # # Plot histograms for each feature of the selected cluster
    # plt.rcParams["figure.figsize"] = (20, 3)
    # for feature in cluster_df.drop(['Cluster'], axis=1):
    #     fig, ax = plt.subplots()
    #     sns.histplot(cluster_df[feature], kde=True, ax=ax)  # Use seaborn for better visuals
    #     ax.set_title(f'Distribution of {feature}')
    #     st.pyplot(fig)
