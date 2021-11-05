import numpy

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor



# configuring the page and the logo
st.set_page_config(page_title='Mohamed Gabr - House Price Prediction', page_icon ='logo.png', layout = 'wide', initial_sidebar_state = 'auto')

st.set_option('deprecation.showPyplotGlobalUse', False)

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
This app predicts the **Prices** of houses in Boston
""")


st.subheader('How to use the model?')
'''
You can use the model by modifying the User Input Parameters on the left. The parameters will be passed to the classification
model and the model will run each time you modify the parameters.

1- You will see the values of the features/ parameters in the **'User Input Parameters'** section in the table below.

2- You will see the prediction result (median value of owner-occupied homes in \$1000s) under the **'Prediction of MEDV'** section below.

4- You will also understand the contribution (weight) of each parameter in the charts shown below

'''

st.subheader('The parameters in the sidebar can be described as below:')
st.write("""
crim
per capita crime rate by town.

zn ==>
proportion of residential land zoned for lots over 25,000 sq.ft.

indus ==>
proportion of non-retail business acres per town.

chas ==>
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

nox ==>
nitrogen oxides concentration (parts per 10 million).

rm ==>
average number of rooms per dwelling.

age ==>
proportion of owner-occupied units built prior to 1940.

dis ==>
weighted mean of distances to five Boston employment centres.

rad ==>
index of accessibility to radial highways.

tax ==>
full-value property-tax rate per \$10,000.

ptratio ==>
pupil-teacher ratio by town.

black ==>
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

lstat ==>
lower status of the population (percent).

medv ==>
median value of owner-occupied homes in \$1000s.
""")

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header("""User input features/ parameters: 

Select/ modify the combination of features below to predict the price
                """)

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
    PTRATIO = numpy.float(st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), float(X.PTRATIO.mean())))
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
st.header('User Input parameters')
st.write(df)

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction (MEDV)')
# st.write(prediction)
html_str = f"""
<h3 style="color:lightgreen;">{prediction} Thousand Dollars $</h3>
"""

st.markdown(html_str, unsafe_allow_html=True)
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

st.info("""**Note: ** [The data source is]: ** (https://www.kaggle.com/c/boston-housing). the following steps have been applied till we reached the model:

        1- Data Acquisition/ Data Collection (reading data, adding headers)

        2- Data Cleaning / Data Wrangling / Data Pre-processing (handling missing values, correcting data fromat/ data standardization 
        or transformation/ data normalization/ data binning/ Preparing Indicator or binary or dummy variables for Regression Analysis/ 
        Saving the dataframe as ".csv" after Data Cleaning & Wrangling)

        3- Exploratory Data Analysis (Analyzing Individual Feature Patterns using Visualizations/ Descriptive statistical Analysis/ 
        Basics of Grouping/ Correlation for continuous numerical variables/ Analysis of Variance-ANOVA for ctaegorical or nominal or 
        ordinal variables/ What are the important variables that will be used in the model?)

        4- Model Development (Single Linear Regression and Multiple Linear Regression Models/ Model Evaluation using Visualization)

        5- Polynomial Regression Using Pipelines (one-dimensional polynomial regession/ multi-dimensional or multivariate polynomial 
        regession/ Pipeline : Simplifying the code and the steps)

        6- Evaluating the model numerically: Measures for in-sample evaluation (Model 1: Simple Linear Regression/ 
        Model 2: Multiple Linear Regression/ Model 3: Polynomial Fit)

        7- Predicting and Decision Making (Prediction/ Decision Making: Determining a Good Model Fit)

        8- Model Evaluation and Refinement (Model Evaluation/ cross-validation score/ over-fitting, under-fitting and model selection)

""")


with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Published By: <a href="https://golytics.github.io/" target="_blank">Dr. Mohamed Gabr</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

