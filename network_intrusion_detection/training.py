import os.path
import sys
from typing import Literal

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

import metrics
import preprocessing

plots_dir = os.path.join(os.path.dirname(__file__), 'plots')


def train_decision_tree(train: DataFrame, split_type: str) -> DecisionTreeClassifier:
    X = train.drop(['Label'], axis='columns')
    y = train['Label']

    decision_tree = DecisionTreeClassifier(max_depth=10, random_state=preprocessing.RANDOM_STATE)
    decision_tree.fit(X, y)

    outfile = os.path.join(plots_dir, f'decision_tree-{split_type}-features.png')
    features_mask = decision_tree.feature_importances_ > 0.04
    metrics.plot_feature_importance(decision_tree.feature_importances_[features_mask],
                                    X.columns[features_mask],
                                    "Decision Tree", outfile)

    return decision_tree


def train_random_forest(train: DataFrame, split_type: str) -> RandomForestClassifier:
    X = train.drop(['Label'], axis='columns')
    y = train['Label']

    random_forest = RandomForestClassifier(max_depth=10, n_jobs=4, max_features=10,
                                           random_state=preprocessing.RANDOM_STATE)
    random_forest.fit(X, y)

    outfile = os.path.join(plots_dir, f'random_forest-{split_type}-features.png')
    features_mask = random_forest.feature_importances_ > 0.04
    metrics.plot_feature_importance(random_forest.feature_importances_[features_mask],
                                    X.columns[features_mask],
                                    "Random Forest", outfile)

    return random_forest


def test_model(which: Literal['decision_tree', 'random_forest'], splitmode: float):
    train, test = preprocessing.get_dataset('MachineLearningCVE', splitmode)
    split_type = '60_40' if 0 < splitmode < 1 else 'days'
    model = train_decision_tree(train, split_type) if which == 'decision_tree' else train_random_forest(train, split_type)

    X_test = test.drop(['Label'], axis='columns')
    y_true = test['Label']
    y_pred = model.predict(X_test)

    print(classification_report(preprocessing.label_encoder.inverse_transform(y_true),
                                preprocessing.label_encoder.inverse_transform(y_pred),
                                labels=preprocessing.label_encoder.classes_, digits=4))
    outfile = os.path.join(plots_dir, f'{which}-{split_type}-conf_matrix.png')
    metrics.plot_confusion_matrix(preprocessing.label_encoder.inverse_transform(y_true),
                                  preprocessing.label_encoder.inverse_transform(y_pred),
                                  preprocessing.label_encoder.classes_,
                                  "Decision Tree" if which == 'decision_tree' else "Random Forest",
                                  outfile)


def main(which: Literal['tree', 'forest', 'all'] = 'all'):
    if which == 'tree' or which == 'all':
        print('\nTraining Decision Tree with 60:40 split...')
        test_model('decision_tree', 0.6)
        print('\nTraining Decision Tree with day split...')
        test_model('decision_tree', -1)
    if which == 'forest' or which == 'all':
        print('\nTraining Random Forest with 60:40 split...')
        test_model('random_forest', 0.6)
        print('\nTraining Random Forest with day split...')
        test_model('random_forest', -1)


if __name__ == '__main__':
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    main('all' if len(sys.argv) < 2 else sys.argv[1])