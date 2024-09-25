import seaborn as sb
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score

from network_intrusion_detection.preprocessing import get_dataset


def plot_confusion_matrix(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    ax = sb.heatmap(m, annot=True, cmap='Blues')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    plt.show()


def train_decision_tree(train: DataFrame) -> DecisionTreeClassifier:
    y = train['Label']
    X = train.drop(['Label'], axis='columns')

    decision_tree = DecisionTreeClassifier(max_depth=10)
    decision_tree.fit(X, y)
    return decision_tree


def test_model():
    train, test = get_dataset('MachineLearningCVE', 0.6)
    decision_tree = train_decision_tree(train)

    X_test = test.drop(['Label'], axis='columns')
    y_true = test['Label']
    y_pred = decision_tree.predict(X_test)

    #print(decision_tree.score(X_test, y_true))
    print(classification_report(y_true, y_pred))
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print(f'Recall: {recall_score(y_true, y_pred)}')
    print(f'F1-Score: {f1_score(y_true, y_pred)}')
    plot_confusion_matrix(y_true, y_pred)


def main():
    test_model()


if __name__ == '__main__':
    main()