from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from ..visualization import generate_graph_tree
from .Translator import Translator
from .Rule import Rule
from .Pattern import Pattern
import numpy as np


class DecisionTree:
    """Class that contains a model created by a decision tree and provides functions over it"""

    def __init__(self, data, label, min_percentage_label_leaf=0.1, max_depth=5, translator=Translator()):

        """Initializer for DecisionTree"""
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
        self.translator = translator

        # Train model
        self.__model.fit(data, label)

        # Generate graph
        self.__generate_graph()

    @property
    def feature_importances(self):
        return self.__model.feature_importances_

    @property
    def tree(self):
        return self.__model.tree_

    def cross_val_score(self, scoring, cv=10):
        return np.mean(cross_val_score(self.__model, self.data, self.label, scoring=scoring, cv=cv, n_jobs=-1))

    def get_patterns(self, max_impurity=0.5, min_sample_size=0):
        """
        Function that return a list of Patterns. It allows to filter by impurity a sample size

        :param max_impurity: Maximum impurity of the pattern (0 - 0.5)
        :param min_sample_size: Minimum percentage of samples with regard to the number of records of the dataset
        :return:  List of Pattern objects
        """
        n_nodes = self.tree.node_count

        if n_nodes == 1:
            raise ValueError("The decision tree only consists of a root node. Patterns can't be defined")

        children_left = self.tree.children_left
        children_right = self.tree.children_right
        total_neg = self.tree.value[0][0][0]
        total_pos = self.tree.value[0][0][1]

        are_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = []
        decisions = []

        self.__get_decisions(children_left, children_right, decisions, stack, 0, are_leaves)
        for id_node, node in enumerate(decisions):
            for idx_decision, decision in enumerate(decisions[id_node]):
                if 'Feature' in decisions[id_node][idx_decision]:
                    feature_idx = decisions[id_node][idx_decision]['Feature']
                    if isinstance(feature_idx, int):
                        value = self.data.columns.values[feature_idx]
                        decisions[id_node][idx_decision]['Feature'] = value
        leaves = []
        for idx, leaf in enumerate(are_leaves):
            if leaf:
                leaves.append(decisions[idx])
        patterns = []
        # Last element contains the leaf information
        valid_leaves = [leaf for leaf in leaves if leaf[-1]['Class'] == 'Positive']
        for leaf in valid_leaves:
            rules = [Rule(node["Feature"], node["Operator"], node["Threshold"], self.translator) for node in
                     leaf[:-1]]
            pattern = Pattern(rules, total_pos, total_neg, leaf[-1]["Impurity"], leaf[-1]["Number_Pos"],
                              leaf[-1]["Number_Neg"])
            if pattern.sample_size > min_sample_size and pattern.impurity < max_impurity:
                patterns.append(pattern)

        return patterns

    def __generate_graph(self):
        label_name = self.label.name.replace('_', ' ')
        binary_labels = ["No " + label_name, label_name]
        self.graph = generate_graph_tree(self.__model, self.data.columns, binary_labels)

    def __get_decisions(self, children_left, children_right, decisions, stack, node_id, are_leaves):
        """
        Recursive function that insert in a list all the decisions of every node in the tree
        
        :param children_left: Node ID of the left children of the node
        :param children_right: Node ID of the right children of the node
        :param decisions: List that will store the decisions of every node in its corresponding index
        :param stack: List that acts like a stock to store the decisions
        :param node_id: Node ID (0 if the node is the root node)
        :param are_leaves: Boolean list that indicates if a node is a leaf
        """
        feature = self.tree.feature
        threshold = self.tree.threshold
        impurity = self.tree.impurity
        samples = self.tree.n_node_samples
        value = self.tree.value

        left_node = children_left[node_id]
        right_node = children_right[node_id]
        decisions.insert(node_id, stack)
        value_neg = value[node_id][0][0]
        value_pos = value[node_id][0][1]
        class_str = 'Negative' if value_neg > value_pos else 'Positive'
        if left_node != right_node:
            stack_left = stack + [
                {'Feature': int(feature[node_id]), 'Operator': '<=', 'Threshold': threshold[node_id],
                 'Impurity': impurity[node_id], 'Samples': samples[node_id], 'Number_Neg': value[node_id][0][0],
                 'Number_Pos': value[node_id][0][1], 'Class': class_str}
            ]
            stack_right = stack + [
                {'Feature': int(feature[node_id]), 'Operator': '>', 'Threshold': threshold[node_id],
                 'Impurity': impurity[node_id], 'Samples': samples[node_id], 'Number_Neg': value[node_id][0][0],
                 'Number_Pos': value[node_id][0][1], 'Class': class_str}
            ]
            self.__get_decisions(children_left, children_right, decisions, stack_left, left_node, are_leaves)
            self.__get_decisions(children_left, children_right, decisions, stack_right, right_node, are_leaves)
        else:
            are_leaves[node_id] = True
            stack += [
                {'Impurity': impurity[node_id], 'Samples': samples[node_id], 'Number_Neg': value_neg,
                 'Number_Pos': value_pos, 'Class': class_str}
            ]
            decisions.insert(node_id, stack)
