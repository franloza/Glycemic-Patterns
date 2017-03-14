from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from visualization import generate_graph_tree
from numpy import mean

class DecisionTree:
    """Class that contains a model created by a decision tree and provides functions over it"""

    def __init__(self, data, label, min_percentage_label_leaf=0.1, max_depth=5):

        """Constructor for DecisionTree"""
        if data.empty:
            raise ValueError("data cannot be empty")
        if len(label.shape) > 1 and label.shape[1] > 1:
            raise ValueError("label argument must contain only one column")

        min_samples = int(min_percentage_label_leaf * len(label))
        self.data = data
        self.label = label

        self.__model = DecisionTreeClassifier(criterion='gini', splitter='best',
                                              max_depth=max_depth,
                                              min_samples_split=2,
                                              min_samples_leaf=min_samples,
                                              min_weight_fraction_leaf=0.0,
                                              max_features=None,
                                              random_state=None,
                                              max_leaf_nodes=None,
                                              min_impurity_split=1e-07,
                                              class_weight=None,
                                              presort=True)

        # Train model
        self.__model.fit(data, label)

        # Generate graph
        self.__generate_graph()

    @property
    def feature_importances(self):
        return self.__model.feature_importances_

    @property
    def graph(self):
        return self.__graph

    @property
    def tree(self):
        return self.__model.tree_

    def cross_val_score(self, scoring, cv=10):
        return mean(cross_val_score(self.__model, self.data, self.label, scoring=scoring, cv=cv, n_jobs=-1))

    def __generate_graph(self):
        label_name = self.label.name.replace('_',' ')
        binary_labels = ["No " + label_name, label_name]
        self.__graph = generate_graph_tree(self.__model, self.data.columns, binary_labels)
