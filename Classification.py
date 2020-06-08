"""
Classification:

1. Algorithms:
    * Logistic Regression
    * Random Forest
    * Support Vector Machine
    * Decision Tree

2. Evaluation:
    * Accuracy score
    * Precision
    * Recall
    * F1 score

3. Hold-out: 75% train, 25% test

4. k-fold cross-validation: not done yet

"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

class Classification:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def logistic_regression(self):
        log_reg = LogisticRegression(solver='liblinear', random_state=0)
        log_reg.fit(self.X_train, self.y_train)
        # save model
        dump(log_reg, 'LogisticRegression.joblib')
        y_pred_test = log_reg.predict(self.X_test)
        y_pred_train = log_reg.predict(self.X_train)
        print('Logistic: Accuracy score on test set: {0:0.4f}'.format(accuracy_score(self.y_test, y_pred_test)))
        print('Logistic: Accuracy score on training set: {0:0.4f}'.format(accuracy_score(self.y_train, y_pred_train)))

    def random_forest(self):
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
        clf_rf.fit(self.X_train, self.y_train)
        # save model
        dump(clf_rf, 'RandomForest.joblib')
        y_pred_test = clf_rf.predict(self.X_test)
        y_pred_train = clf_rf.predict(self.X_train)
        print('Random forest: Accuracy score on test set: {0:0.4f}'.format(accuracy_score(self.y_test, y_pred_test)))
        print('Random forest: Accuracy score on training set: {0:0.4f}'.format(accuracy_score(self.y_train, y_pred_train)))

    def support_vector_machine(self):
        clf_svm = svm.SVC()
        clf_svm.fit(self.X_train, self.y_train)
        # save model
        dump(clf_svm, 'SupportVectorMachine.joblib')
        y_pred_test = clf_svm.predict(self.X_test)
        y_pred_train = clf_svm.predict(self.X_train)
        print('SVM: Accuracy score on test set: {0:0.4f}'.format(accuracy_score(self.y_test, y_pred_test)))
        print('SVM: Accuracy score on training set: {0:0.4f}'.format(accuracy_score(self.y_train, y_pred_train)))

    def decisiontree(self):
        clf_dt = DecisionTreeClassifier(random_state=0)
        clf_dt.fit(self.X_train, self.y_train)
        # save model
        dump(clf_dt, 'DecisionTree.joblib')
        y_pred_test = clf_dt.predict(self.X_test)
        y_pred_train = clf_dt.predict(self.X_train)
        print('Decision Tree: Accuracy score on test set: {0:0.4f}'.format(accuracy_score(self.y_test, y_pred_test)))
        print('Decision Tree: Accuracy score on training set: {0:0.4f}'.format(accuracy_score(self.y_train, y_pred_train)))