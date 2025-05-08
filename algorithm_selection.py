import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import random

# Decision Tree Classifier Prediction
def predict_decision_tree(train_data, test_data):
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    train_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    test_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    train_data_vals = train_data[train_data.columns.difference(['problem', 'dim',"best_algorithms"])].values
    test_data_vals = test_data[test_data.columns.difference(['problem', 'dim',"best_algorithms"])].values
    train_data_vals = train_data_vals.astype(np.float32)
    test_data_vals = test_data_vals.astype(np.float32)

    clf = DecisionTreeClassifier().fit(np.array(train_data_vals), np.array(train_data["best_algorithms"].tolist()))
    y_pred = clf.predict(np.array(test_data_vals))
    acc_precision = AS_Precision(test_data["best_algorithms"].tolist(), y_pred.tolist())

    return acc_precision

# Random Forest Classifier Prediction
def predict_random_forest(train_data, test_data):
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    train_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    test_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    train_data_vals = train_data[train_data.columns.difference(['problem', 'dim',"best_algorithms"])].values
    test_data_vals = test_data[test_data.columns.difference(['problem', 'dim',"best_algorithms"])].values
    train_data_vals = train_data_vals.astype(np.float32)
    test_data_vals = test_data_vals.astype(np.float32)

    clf = RandomForestClassifier().fit(np.array(train_data_vals), np.array(train_data["best_algorithms"].tolist()))
    y_pred = clf.predict(np.array(test_data_vals))
    acc_precision = AS_Precision(test_data["best_algorithms"].tolist(), y_pred.tolist())

    return acc_precision

# k-Nearest Neighbors Classifier Prediction
def predict_knn(train_data, test_data):
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    train_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    test_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    train_data_vals = train_data[train_data.columns.difference(['problem', 'dim',"best_algorithms"])].values
    test_data_vals = test_data[test_data.columns.difference(['problem', 'dim',"best_algorithms"])].values
    train_data_vals = train_data_vals.astype(np.float32)
    test_data_vals = test_data_vals.astype(np.float32)

    clf = KNeighborsClassifier().fit(np.array(train_data_vals), np.array(train_data["best_algorithms"].tolist()))
    y_pred = clf.predict(np.array(test_data_vals))
    acc_precision = AS_Precision(test_data["best_algorithms"].tolist(), y_pred.tolist())

    return acc_precision

# Dummy Classifier Prediction
# Uses most frequent strategy for baseline comparison
def predict_dummy(train_data, test_data):
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    train_data.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
    test_data.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
    train_data_vals = train_data[train_data.columns.difference(['problem', 'dim',"best_algorithms"])].values
    test_data_vals = test_data[test_data.columns.difference(['problem', 'dim', "best_algorithms"])].values

    train_y = np.array(train_data["best_algorithms"].tolist())
    test_y = np.array(test_data["best_algorithms"].tolist())
    y_result = []
    for i in range(test_y.shape[1]):
        clf = DummyClassifier(strategy="most_frequent").fit(train_data_vals, train_y[:,i].tolist())
        y_result.append(clf.predict(test_data_vals))
    y_result = np.array(y_result).transpose()

    acc_precision = AS_Precision(test_y, y_result)

    return acc_precision

# Data Preprocessing Function
# Handles missing, infinite values and extracts relevant features
def preprocess_features(df):
    # The constrained Pareto front is equal to 0
    df.loc[:,'ps_dist_max'] = df.loc[:,'ps_dist_max'].replace(np.nan, 0)
    df.loc[:,'ps_dist_mean'] = df.loc[:,'ps_dist_mean'].replace(np.nan, 0)
    df.loc[:,'ps_dist_iqr_mean'] = df.loc[:,'ps_dist_iqr_mean'].replace(np.nan, 0)
    df.loc[:,'pf_dist_max'] = df.loc[:,'pf_dist_max'].replace(np.nan, 0)
    df.loc[:,'pf_dist_mean'] = df.loc[:,'pf_dist_mean'].replace(np.nan, 0)
    df.loc[:,'pf_dist_iqr_mean'] = df.loc[:,'pf_dist_iqr_mean'].replace(np.nan, 0)

    df.loc[:,'hv_uhv_n'] = df.loc[:,'hv_uhv_n'].replace([np.inf, -np.inf], 0)
    df.loc[:,'gd_cpo_upo'] = df.loc[:,'gd_cpo_upo'].replace(np.inf, 999999)
    df.loc[:,'corr_cf'] = df.loc[:,'corr_cf'].replace(np.nan, 0)
    # The Pearson's correlation coefficient is not defined
    df.loc[:,'nuhv_r1_rws'] = df.loc[:,'nuhv_r1_rws'].replace(np.nan, 0)
    df.loc[:,'ncv_r1_rws'] = df.loc[:,'ncv_r1_rws'].replace(np.nan, 0)
    df.loc[:,'nncv_r1_rws'] = df.loc[:,'nncv_r1_rws'].replace(np.nan, 0)
    df.loc[:,'sup_r1_rws'] = df.loc[:,'sup_r1_rws'].replace(np.nan, 0)
    df.loc[:,'lnd_r1_rws'] = df.loc[:,'lnd_r1_rws'].replace(np.nan, 0)
    df.loc[:,'nhv_r1_rws'] = df.loc[:,'nhv_r1_rws'].replace(np.nan, 0)
    df.loc[:,'bhv_r1_rws'] = df.loc[:,'bhv_r1_rws'].replace(np.nan, 0)
    df.loc[:,'dist_c_r1_rws'] = df.loc[:,'dist_c_r1_rws'].replace(np.nan, 0)
    df.loc[:,'inf_r1_rws'] = df.loc[:,'inf_r1_rws'].replace(np.nan, 0)
    df.loc[:,'inc_r1_rws'] = df.loc[:,'inc_r1_rws'].replace(np.nan, 0)
    df.loc[:,'nfronts_r1_rws'] = df.loc[:,'nfronts_r1_rws'].replace(np.nan, 0)
    df.loc[:,'dist_c_dist_x_avg_r1'] = df.loc[:,'dist_c_dist_x_avg_r1'].replace(np.nan, 0)
    # All solutions have equal uncontrained rank
    df.loc[:,'skew_f'] = df.loc[:,'skew_f'].replace(np.nan, 0)
    df.loc[:,'kurt_f'] = df.loc[:,'kurt_f'].replace(np.nan, df.loc[:,'kurt_f'].mean())
    # print(df.columns[df.isna().any()].tolist())
    for column in df.columns:
        if df[column].isin([np.inf, -np.inf]).values.any():
            print(column)
    # print(df.describe())
    df = df.fillna(-1)

    return df

# AS_Precision Calculation
def AS_Precision(true, predicted):
    result = []
    for i in range(len(true)):
        TP = np.sum(np.logical_and(true[i], predicted[i]))
        TP_FP = np.sum(predicted[i])
        if TP_FP == 0:
            result.append(0)
        else:
            result.append(TP/TP_FP)
    return np.average(result)

# Main Leave-One-Problem-Out (LOPO) Evaluation Function
def main_LOPO(feature_sel=False):
    random.seed(10)
    dim = 10
    cut = '1'

    df = pd.read_csv("data/ELA_features.csv")
    df = df[df['dim']==dim]
    targets1 = pd.read_csv("data/best_algorithms_"+cut+"cut_"+str(dim)+"d.csv")
    targets1['problems'] = targets1['problems'].str.replace('-', '')
    df = df.merge(targets1, left_on='problem', right_on='problems', how='inner')
    df = df[df['dim']==dim]
    df = df.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'problems'], axis=1)

    # Multi-label binarization
    best_algs = [[tmp.split("_")[0] for tmp in elem.split(' ')] for elem in df['best_algorithms'].tolist()]
    mlb = MultiLabelBinarizer()
    best_algs_coded = mlb.fit_transform(best_algs)

    df['best_algorithms'] = best_algs_coded.tolist()

    problem_count = 0
    accs_dummy, accs_dt, accs_rf, accs_knn = [], [], [], []

    problems = list(set(targets1['problems'].tolist()))
    problems.sort()

    # LOPO Evaluation
    for problem in problems:
        print(problem)
        problem_count += 1
        test_data = df[df['problem']==problem]
        train_data = df[(df['problem']!=problem)]
        acc_dummy = predict_dummy(train_data, test_data)
        acc_dt = predict_decision_tree(train_data, test_data)
        acc_rf = predict_random_forest(train_data, test_data)
        acc_knn = predict_knn(train_data, test_data)
        accs_dummy.append(acc_dummy)
        accs_dt.append(acc_dt)
        accs_rf.append(acc_rf)
        accs_knn.append(acc_knn)

    print("Dummy model:")
    print(np.mean(accs_dummy))
    print("Decision tree")
    print(np.mean(accs_dt))
    print("Random forest")
    print(np.mean(accs_rf))
    print("kNN")
    print(np.mean(accs_knn))

if __name__ == "__main__":
    main_LOPO()