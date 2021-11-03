import numpy

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor



# configuring the page and the logo
st.set_page_config(page_title='Mohamed Gabr - House Price Prediction', page_icon ='logo.png', layout = 'wide', initial_sidebar_state = 'auto')


import os
import base64

# the functions to prepare the image to be a hyperlink
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


# preparing the layout for the top section of the app
# dividing the layout vertically (dividing the first row)
row1_1, row1_2, row1_3 = st.columns((1, 4, 5))

# first row first column
with row1_1:
    gif_html = get_img_with_href('logo.png', 'https://golytics.github.io/')
    st.markdown(gif_html, unsafe_allow_html=True)

with row1_2:
    # st.image('logo.png')
    st.title("Predicting House Prices in Boston, USA")
    st.markdown("<h2>A POC for a Real Estate Client</h2>", unsafe_allow_html=True)

# first row second column
with row1_3:
    st.info(
        """
        ##
        This data product has been prepared as a proof of concept of a machine learning model to predict prices of houses in Cairo, Egypt.
        For demonstration purposes, we have used the data from Boston to prove the technical feasibility of the model. Developing the final model required
        many steps following the CRISP-DM methodology. After building the model we used it to predict the prices in this application. **The model can be changed/
        enhanced for any another city based on its own data.**
        """)
st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = numpy.float(st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), float(X.CRIM.mean())))
    ZN = numpy.float(st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), float(X.ZN.mean())))
    INDUS = numpy.float(st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), float(X.INDUS.mean())))
    CHAS = numpy.float(st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), float(X.CHAS.mean())))
    NOX = numpy.float(st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), float(X.NOX.mean())))
    RM = numpy.float(st.sidebar.slider('RM', X.RM.min(), X.RM.max(), float(X.RM.mean())))
    AGE = numpy.float(st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), float(X.AGE.mean())))
    DIS = numpy.float(st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), float(X.DIS.mean())))
    RAD = numpy.float(st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), float(X.RAD.mean())))
    TAX = numpy.float(st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), float(X.TAX.mean())))
    PTRATIO = numpy.float(st.sidebar.slider('PTRATIO', X.PTRATIO.min(), float(X.PTRATIO.max(), X.PTRATIO.mean())))
    B = numpy.float(st.sidebar.slider('B', X.B.min(), X.B.max(), float(X.B.mean())))
    LSTAT = numpy.float(st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), float(X.LSTAT.mean())))
    data = {'CRIM': float(CRIM),
            'ZN': float(ZN),
            'INDUS': float(INDUS),
            'CHAS': float(CHAS),
            'NOX': float(NOX),
            'RM': float(RM),
            'AGE': float(AGE),
            'DIS': float(DIS),
            'RAD': float(RAD),
            'TAX': float(TAX),
            'PTRATIO': float(PTRATIO),
            'B': B,
            'LSTAT': float(LSTAT)}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance (weight) based on the values shown in the chart')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance (weight) based on the values shown in the bar chart')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')


with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)