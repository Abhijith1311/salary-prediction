import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
st.title("Salary predition")
data=pd.read_csv("Salary_data.csv")
X=np.array(data["No.of years"]).reshape(-1,1)
y=np.array(data["salary"]).reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
lr=LinearRegression()
lr.fit(X_train,y_train)

nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav =="Home":
    st.image("salary vs years.jpg")
    if st.checkbox("Show Table" ):
        st.table(data)
    v1=st.slider("years",0,20)
    data=data.loc[data["No.of years"]>=v1]
    v2=st.number_input("years",0,20)
    data=data.loc[data["No.of years"]>=v2]


    plt.figure(figsize=(10,5))
    plt.scatter(data["No.of years"],data["salary"])
    plt.ylim(0)
    x1=plt.xlabel("No.of years")
    y1=plt.ylabel("Salary")
    plt.tight_layout()
    st.pyplot(plt)


if nav =="Prediction":



    st.header("know your salary")
    val=st.number_input("Enter your experience")
    val=np.array(val).reshape(1,-1)
    pred=lr.predict(val)[0]
    if st.button("Predict"):  
        st.write(pred)
        st.write('Prediction')
if nav =="Contribute":
    st.header("contribute to our dataset")
    ex=st.number_input("enter your experience",0,20)
    sal=st.number_input("enter your salary",30000.00,1000000.00)
    if st.button("submit"):
        to_add={"No.of years":[ex],"salary":[sal]}
        to_add=pd.DataFrame(to_add)
        to_add.to_csv("Salary_Data.csv",mode='a',header=False,index=False)
        st.success("Submitted")
