from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, \
    roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler

import numpy as np
from tqdm import tqdm


# 获取数据
def get_data():
    '''
    需要考虑是否需要做 数据均衡

    :return:
    '''
    data = np.load('../data/dataset.npy')
    m, n = data.shape
    x = data[:, 0:n - 1]
    y = data[:, -1]

    ros = RandomOverSampler(random_state=0)
    x_b, y_b = ros.fit_resample(x, y)
    return x_b, y_b


# LR
def SK_LR(data_x, data_y):
    log = open('../data/result/LR.txt', 'a', encoding='UTF-8')
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, train_size=0.6)
    tunned_parameters = {
        'C': [1e-1, 1, 10]
    }
    scores = ['precision']
    for score in scores:
        clf = GridSearchCV(LogisticRegression(), tunned_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(data_x_train, data_y_train)

        print("Best parameters set found on development set:", file=log)
        print(clf.best_params_, file=log)
        y_true, y_pred = data_y_test, clf.predict(data_x_test)
        print(classification_report(y_true, y_pred, digits=4), file=log)
        print('混淆矩阵：', file=log)
        print(confusion_matrix(y_true, y_pred), file=log)
        print('准确率：', accuracy_score(y_true, y_pred), file=log)
        print('错误率：', 1 - accuracy_score(y_true, y_pred), file=log)
        print('精准率：', precision_score(y_true, y_pred), file=log)
        print('F1值：', f1_score(y_true, y_pred), file=log)
        print('roc-auc：', roc_auc_score(y_true, y_pred), file=log)


def SK_SVM(data_x, data_y):
    log = open('../data/result/SVM.txt', 'w', encoding='UTF-8')
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, train_size=0.6)
    tunned_parameters = {
        'C': [1e-1, 1, 10],
        'gamma': [0.001, 0.0001]
    }
    scores = ['precision']
    for score in scores:
        clf = GridSearchCV(SVC(), tunned_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(data_x_train, data_y_train)

        print("Best parameters set found on development set:", file=log)
        print(clf.best_params_, file=log)
        y_true, y_pred = data_y_test, clf.predict(data_x_test)
        print(classification_report(y_true, y_pred, digits=4), file=log)
        print('混淆矩阵：', file=log)
        print(confusion_matrix(y_true, y_pred), file=log)
        print('准确率：', accuracy_score(y_true, y_pred), file=log)
        print('错误率：', 1 - accuracy_score(y_true, y_pred), file=log)
        print('精准率：', precision_score(y_true, y_pred), file=log)
        print('F1值：', f1_score(y_true, y_pred), file=log)
        print('roc-auc：', roc_auc_score(y_true, y_pred), file=log)


# 随机森林
def SK_RF(data_x, data_y):
    log = open('../data/result/RF.txt', 'w', encoding='UTF-8')
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, train_size=0.6)
    tunned_parameters = {
        'n_estimators': [10, 20, 30],
        'max_depth': [3, 5, 7],
    }
    scores = ['precision']
    for score in scores:
        clf = GridSearchCV(RandomForestClassifier(), tunned_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(data_x_train, data_y_train)

        print("Best parameters set found on development set:", file=log)
        print(clf.best_params_, file=log)
        y_true, y_pred = data_y_test, clf.predict(data_x_test)
        print(classification_report(y_true, y_pred, digits=4), file=log)
        print('混淆矩阵：', file=log)
        print(confusion_matrix(y_true, y_pred), file=log)
        print('准确率：', accuracy_score(y_true, y_pred), file=log)
        print('错误率：', 1 - accuracy_score(y_true, y_pred), file=log)
        print('精准率：', precision_score(y_true, y_pred), file=log)
        print('F1值：', f1_score(y_true, y_pred), file=log)
        print('roc-auc：', roc_auc_score(y_true, y_pred), file=log)


# DNN
def SK_DNN(data_x, data_y):
    log = open('../data/result/DNN.txt', 'w', encoding='UTF-8')
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, train_size=0.6)
    tunned_parameters = {
        'activation': ['identity', 'logistic'],
        'hidden_layer_sizes': [(256, 256), (256, 128), (256, 64)]
    }
    scores = ['precision']
    for score in scores:
        clf = GridSearchCV(MLPClassifier(max_iter=5000), tunned_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(data_x_train, data_y_train)

        print("Best parameters set found on development set:", file=log)
        print(clf.best_params_, file=log)
        y_true, y_pred = data_y_test, clf.predict(data_x_test)
        print(classification_report(y_true, y_pred, digits=4), file=log)
        print('混淆矩阵：', file=log)
        print(confusion_matrix(y_true, y_pred), file=log)
        print('准确率：', accuracy_score(y_true, y_pred), file=log)
        print('错误率：', 1 - accuracy_score(y_true, y_pred), file=log)
        print('精准率：', precision_score(y_true, y_pred), file=log)
        print('F1值：', f1_score(y_true, y_pred), file=log)
        print('roc-auc：', roc_auc_score(y_true, y_pred), file=log)


if __name__ == '__main__':
    data_x, data_y = get_data()

    for i in tqdm(range(10)):
        SK_LR(data_x, data_y)
        SK_SVM(data_x, data_y)
        SK_RF(data_x, data_y)
        SK_DNN(data_x, data_y)
