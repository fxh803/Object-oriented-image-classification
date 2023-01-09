from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn import svm


def classification_kNN(n_segments, features, trainmat, label, K, mode="uniform"):
    # 设定knn算法
    neigh = kNN(n_neighbors=K, algorithm="auto", weights=mode, n_jobs=1)
    neigh.fit(trainmat, label)

    # 每个对象的分类后的类型
    type = []
    for i in range(n_segments):
        testmat = [features[i]]
        type.append(neigh.predict(testmat))
    return type


def classification_SVM(n_segments, features, trainmat, label):
    # 设定svm算法
    clf = svm.SVC(C=1, kernel="rbf", decision_function_shape="ovo")
    clf.fit(trainmat, label)

    # 每个对象的分类后的类型
    type = []
    for i in range(n_segments):
        testmat = [features[i]]
        type.append(clf.predict(testmat))
    return type
