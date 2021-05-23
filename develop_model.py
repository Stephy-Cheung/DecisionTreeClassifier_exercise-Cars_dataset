import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('data\car.data', names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

# General information
print (df.info())
print (df.describe())

# Univariate analysis -imbalance of target sample
sns.countplot(data = df, x = 'class').set_title('Counts of car classes')
plt.show()

# Bivariate analysis -relationship between independent variable and target variable
independent_col = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
fig, ax = plt.subplots(3,2, figsize = (10,10))
for index, i in enumerate(independent_col): 
    sns.countplot(data=df, x=i, hue = 'class', ax = ax [int(index/2)][index%2])

# Preprecessing
# Train-test-split
y = df['class']
X = df.drop ('class', axis =1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Encode for categorical independent variable with Ordinal Encoder
buying_cat = ['low', 'med', 'high','vhigh']
maint_cat = ['low', 'med', 'high','vhigh']
doors_cat = ['2','3','4', '5more']
persons_cat	= ['2','4','more']
lug_boot_cat = ['small','med', 'big']
safety_cat = ['low','med','high']

ord_enc = OrdinalEncoder(categories = [buying_cat, maint_cat, doors_cat, persons_cat, lug_boot_cat, safety_cat])

X_train = ord_enc.fit_transform(X_train)
X_test = ord_enc.transform(X_test)

# Encode for target variable 
y_train = y_train.apply(lambda x:['unacc', 'acc', 'good', 'vgood'].index(x))
y_test = y_test.apply(lambda x:['unacc', 'acc', 'good', 'vgood'].index(x))

# Build Decision Tree Classifier
tree1 = DecisionTreeClassifier()
tree1.fit(X_train, y_train)
print ('Feature importances: ', tree1.feature_importances_)
print ('Maximum tree depth: ', tree1.get_depth())
print ('Accuracy: ',tree1.score(X_test, y_test))
y_pred = tree1.predict(X_test)
print('Classification Report: ',classification_report(y_test, y_pred))

# Fine tune and visualize model (Not include in Streamlit app)
from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'max_depth': np.arange(3,12,1)}
tree2 = GridSearchCV(tree1, parameters)
tree2.fit(X_train, y_train)

print (tree2.best_score_)
print (tree2.best_params_) # max-depth : 11

# Update model 
tree2 = DecisionTreeClassifier(max_depth = 11)
tree2.fit(X_train, y_train)
y_pred = tree2.predict(X_test)
print('Classification Report: ',classification_report(y_test, y_pred))

# Visualize the tree 
# In-text
from sklearn import tree
tree_in_text = tree2.export_text(tree1)
print(tree_in_text)

# In-graph
from sklearn.tree import plot_tree
feature_names = df.columns[:6]
target_names = df['class'].unique().tolist()

fig = plt.figure(figsize = (25,20))
plot_tree(tree2, feature_names = feature_names, class_names = target_names, filled = True, rounded = True)

plt.savefig('Tree1.png')