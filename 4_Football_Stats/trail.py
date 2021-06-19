import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import tree
import graphviz

import matplotlib.pyplot as plt
from pdpbox import pdp, get_dataset, info_plots


data = pd.read_csv('statistics.csv')
# print(data.head())

features = [col for col in data.columns if data[col].dtype in [np.int64]]
print(features)

X = data[features]
y = (data['Man of the Match'] == 'Yes')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=23)

feature_names = X_train.columns

model_dt = DecisionTreeClassifier(random_state=22, max_depth=5, min_samples_split=5).fit(X_train, y_train)

# tree_graph = tree.export_graphviz(model_dt, out_file=None, feature_names=feature_names)

# graphviz.Source(tree_graph)

# Plot the pdp for "Goal Scored"

pdp_goals = pdp.pdp_isolate(model=model_dt, dataset=X_valid, model_features=feature_names, feature='Goal Scored')

pdp.pdp_plot(pdp_goals, 'Goal Scored')

plt.show()

