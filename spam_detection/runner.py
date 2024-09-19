import sys

import bayes
import comparison
import knn
import logistic

args = sys.argv

def run_experiment(name: str) -> bool:
    if name == 'bayes':
        print('Running Naive Bayes experiment')
        bayes.compare_sizes()
    elif name == 'knn':
        print('Running K-Nearest Neighbours experiment')
        knn.compare_k_values()
    elif name == 'logistic':
        print('Running Logistic Regression experiment')
        logistic.compare_c_values()
    elif name == 'comparison':
        print('Running Comparison experiment')
        comparison.main()
    else:
        return False
    return True

def print_help():
    print('Usage: runner.py <experiment_name>')
    print('Possible experiment names:')
    print('bayes\t\tRun the Naive Bayes experiment')
    print('knn\t\tRun the K-Nearest Neighbours experiment')
    print('logistic\tRun the Logistic Regression experiment')
    print('comparison\tRun the Comparison experiment')

if len(args) < 2:
    print('\n\nRunning Naive Bayes experiment')
    bayes.compare_sizes()
    print('\n\nRunning K-Nearest Neighbours experiment')
    knn.compare_k_values()
    print('\n\nRunning Logistic Regression experiment')
    logistic.compare_c_values()
    print('\n\nRunning Comparison experiment')
    comparison.main()
elif len(args) == 2:
    if args[1] == 'help':
        print_help()
    elif not run_experiment(args[1]):
        print('Invalid model name', end='\n\n')
        print_help()
else:
    print('Invalid arguments', end='\n\n')
    print_help()