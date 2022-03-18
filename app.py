#Library imports
from email import header
from ssl import Options
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import load_model

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# Loading pickled file.
#file = open('D:\Bank-Customer-churn\Project code and Files\model.pkl', 'rb')
model = pickle.load(open('./pickle_file/model.pkl', 'rb'))
def predict(data):
    df = data.copy()
    df['BalanceSalaryRatio'] = df.Balance/df.EstimatedSalary
    df['TenureByAge'] = df.Tenure/(df.Age)
    df['Gender'].replace({'Male': 1,'Female': 0},inplace=True)
    df1 = pd.get_dummies(data=df, columns=['Geography'])
    scale_var = ['Tenure','CreditScore','Age','Balance','NumOfProducts','EstimatedSalary']
    scaler = MinMaxScaler()
    df1[scale_var] = scaler.fit_transform(df1[scale_var])
    print(df1.columns)
    prediction = model.predict(df1)
    y_pred = []
    for element in prediction:
        if element > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    return y_pred

with header:
    #Setting Title of App
    st.title("Bank Customer Churn Prediction.")
    st.text("Predict if a User is going to churn or not.")
    
with dataset:
    def main():
        st.header("Data Input.")
        st.text("This data is used to predict.")
        html_temp = """
        <style>
            .reportview-container .main .block-container{{
                max-width: 90%;
                padding-top: 5rem;
                padding-right: 5rem;
                padding-left: 5rem;
                padding-bottom: 5rem;
            }}
            img{{
                max-width:40%;
                margin-bottom:40px;
            }}
        </style>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        
        CreditScore = st.number_input("Please Enter credist.selectbotscore",0)
        Geography = 'France','Spain','Germany'
        Gender = st.selectbox("What is sexual orientation?", ["Male","Female"],index=0)
        Age = st.number_input("Please put age here.", 0)
        Tenure = st.number_input("Please put  tenure here.",0)
        Balance = st.number_input("Please put current balace in a account.")
        NumOfProducts = st.number_input("Please put number of products here.",0)
        HasCrCard = st.number_input("Please put 1 if credit card otherwise put 0.",0)
        IsActiveMember = st.number_input("Please put 1 if active member otherwise put 0.",0)
        EstimatedSalary = st.number_input("please put Estimated Salary in integer format.",0)
        
        
        
        input_list = ["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
        data = pd.DataFrame(data=[[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]],columns=input_list)
        
        result=""
        if st.button("Predict"):
            result=predict(data)
            print(result)
            for i in result[i]:
                if i==1:
                    st.success('Customer will churn so keep your service good.')
                else:
                    st.success("Customer will stay with your bank.")
        if st.button("About"):
            st.text("Lets LEarn")
            st.text("Built with Streamlit")
    

if __name__=='__main__':
    main()