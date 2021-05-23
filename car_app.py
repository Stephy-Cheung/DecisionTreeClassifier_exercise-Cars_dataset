import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

@st.cache
def get_data():
    url = 'data/car.data'
    return pd.read_csv(url, names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

# Load data set 
df = get_data ()

st.title('Car acceptance prediction')

st.header('Objective')
st.text('Our goal is to predict the class value of cars.')

st.header('Data at a Glance')
st.dataframe(df.head())

st.header('Data Exploration')
st.text('In this section, we will present interesting obervation of this data set.')

# Bar plot for target variables
st.subheader('Univariate data analysis')
st.text('Imbalance of data - Majority of the cars are classified as \'unacc\' and few cars ')
st.text('are in the class of \'vgood\' and \'good\'.' )
fig, ax =plt.subplots()
sns.countplot(x='class', data =df)
st.pyplot(fig)

# Bar plot for Bivariate data analysis with the target
st.subheader('Bivariate data analysis')
st.text('Cars that can be classified as \'vgood\' or \'good\' normally with a medium or low ')
st.text('buying price and maintenance fee. The cars should be 4-seater or above and with a ')
st.text('medium or big luggage truck, and high safety level. ')
col = ['buying','maint','doors','persons','lug_boot', 'safety']

fig, ax = plt.subplots(3,2, figsize = (10,10))
for index, i in enumerate(col): 
    sns.countplot(data=df, x=i, hue = 'class', ax = ax [int(index/2)][index%2])
st.pyplot(fig)

# Preprocessing
st.header('Data Preprocessing')
st.dataframe (df.head())

st.subheader('Shape of DataFrame and column values')
st.text('Shape of DataFrame: ')
st.text(df.shape)

st.text('Unique column values: ')
for i in df.columns:
    st.write(i, ':' )
    st.text(df[i].unique())

st.subheader('Missing value')
st.text('There is no missing value in the data')
fig, ax = plt.subplots()
ax = sns.heatmap(df.isnull(),cbar =False)
st.pyplot(fig)

# Encode categorical data by ordinal encoder
st.subheader('Encode categorical data into numerical: ')

buying_cat = ['low', 'med', 'high','vhigh'] # 0,1,2,3
maint_cat = ['low', 'med', 'high','vhigh']
doors_cat = ['2','3','4', '5more']
persons_cat	= ['2','4','more']
lug_boot_cat = ['small','med', 'big']
safety_cat = ['low','med','high']
class_cat = ['unacc', 'acc', 'good', 'vgood']

ord_enc = OrdinalEncoder(categories = [buying_cat, maint_cat, doors_cat, persons_cat, lug_boot_cat, safety_cat, class_cat])
df = ord_enc.fit_transform(df) # in numpy.ndarray
df = pd.DataFrame(df, columns =['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

st.dataframe(df.head())

X = df.drop ('class', axis = 1)
y = df['class']


# Sidebar
st.sidebar.title('Desicion Tree Classifier Tuning')
criterion = st.sidebar.selectbox("Select criterion:", ['gini', 'entropy'])
splitter = st.sidebar.selectbox("Select splitter:", ['best', 'random'])
max_depth = st.sidebar.slider('Maximum depth of tree: ', min_value = 3, max_value = 12, step = 1)
test_size = st.sidebar.slider('Test size: ', min_value = 0.1, max_value=0.9, step = 0.1)

st.sidebar.title ('Car Class predictor')
buying_cat = ['Low', 'Medium', 'High','Very high']
maint_cat = ['Low', 'Medium', 'High','Very high']
person_cat = ['2','4','More']
lug_boot_cat = ['Small','Medium', 'Big']
safety_cat = ['Low','Medium','High']

buying = st.sidebar.selectbox('Buying price: ', buying_cat)
maint = st.sidebar.selectbox('Maintainence: ', maint_cat)
door = st.sidebar.slider('Number of doors: (choose \'5\' for 5 doors or above)', min_value=2,max_value=5,step =1)
person = st.sidebar.selectbox('Number of people: ', person_cat )
lug_boot = st.sidebar.selectbox('Luggage boot size: ', lug_boot_cat)
safety = st.sidebar.selectbox('Safety performance: ', safety_cat)

buying = buying_cat.index(buying)
maint = maint_cat.index(maint)
person = person_cat.index(person)
lug_boot = lug_boot_cat.index(lug_boot)
safety = safety_cat.index(safety)

# Build Classifier
st.header('Desicion Tree Classifier')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth = max_depth, random_state=42)
tree.fit(X_train, y_train)
st.subheader("Mean Accuracy")
st.subheader(tree.score(X_test, y_test))

y_predict = tree.predict(X_test)

st.subheader('Feature Importances: ')
st.code(tree.feature_importances_)

st.subheader('Classification Report')
st.code(classification_report(y_test, y_predict))

# Prediction
st.header('Prediction')
test_data = [[buying,maint,door,person,lug_boot,safety]]
prediction = tree.predict(test_data)
if  prediction == 0: 
    st.subheader('This car belongs to class \'unacc\'')
elif prediction == 1: 
    st.subheader('This car belongs to class \'acc\'')
elif prediction == 2: 
    st.subheader('This car belongs to class \'good\'')
elif prediction == 3: 
    st.subheader('This car belongs to class \'very good\'')
