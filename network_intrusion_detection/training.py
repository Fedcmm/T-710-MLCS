from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from network_intrusion_detection.metrics import display_metrics, plot_feature_importance
from network_intrusion_detection.preprocessing import get_dataset


def train_decision_tree(train: DataFrame) -> DecisionTreeClassifier:
    X = train.drop(['Label'], axis='columns')
    y = train['Label']

    decision_tree = DecisionTreeClassifier(max_depth=10)
    decision_tree.fit(X, y)
    return decision_tree


def train_random_forest(train: DataFrame) -> RandomForestClassifier:
    X = train.drop(['Label'], axis='columns')
    y = train['Label']

    random_forest = RandomForestClassifier(max_depth=10, n_jobs=4, max_features=10)
    random_forest.fit(X, y)
    plot_feature_importance(random_forest.feature_importances_[:10], X.columns[:10])
    return random_forest


def test_model(training_function, splitmode):
    train, test = get_dataset('MachineLearningCVE', splitmode)
    model = training_function(train)

    X_test = test.drop(['Label'], axis='columns')
    y_true = test['Label']
    y_pred = model.predict(X_test)

    display_metrics(y_true, y_pred)


def main():
    #test_model(train_decision_tree, 0.6)
    test_model(train_random_forest, 0.6)


if __name__ == '__main__':
    main()