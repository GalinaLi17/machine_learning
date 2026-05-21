import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv('heart.csv')
print('Размер данных:', df.shape)
print('Баланс классов:\n', df['target'].value_counts())

#категор признаки
cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
X = pd.get_dummies(df.drop(columns='target'), columns=cat_cols, drop_first=True)
y = df['target']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#оценка модели
def evaluate(name, model):
    pred = model.predict(Xte)
    if hasattr(model, 'predict_proba'):
        score = model.predict_proba(Xte)[:, 1]
    else:
        score = model.decision_function(Xte)
    print('\n' + name)
    print('Accuracy:', round(accuracy_score(yte, pred), 3))
    print('Precision:', round(precision_score(yte, pred), 3))
    print('Recall:', round(recall_score(yte, pred), 3))
    print('F1-score:', round(f1_score(yte, pred), 3))
    print('ROC-AUC:', round(roc_auc_score(yte, score), 3))
    print('Confusion matrix:\n', confusion_matrix(yte, pred))

#модели с параметрами по умолчанию
log_def = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
svm_def = make_pipeline(StandardScaler(), SVC(kernel='linear')).fit(Xtr, ytr)
tree_def = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
evaluate('LogisticRegression default', log_def)
evaluate('SVC linear default', svm_def)
evaluate('DecisionTree default', tree_def)

#подбор гиперпараметров
grids = {
    'LogisticRegression': (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        {'logisticregression__C':[0.01,0.1,1,10],
         'logisticregression__penalty':['l1','l2'],
         'logisticregression__solver':['liblinear','saga']}
    ),
    'SVC': (
        make_pipeline(StandardScaler(), SVC()),
        {'svc__C':[0.1,1,10,100],
         'svc__gamma':['scale','auto',0.01,0.1],
         'svc__kernel':['rbf','poly']}
    ),
    'DecisionTree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth':[3,5,10,None],
         'min_samples_split':[2,5,10],
         'criterion':['gini','entropy']}
    )
}
best = {}
for name, (model, params) in grids.items():
    search = GridSearchCV(model, params, cv=5, scoring='roc_auc')
    search.fit(Xtr, ytr)
    best[name] = search.best_estimator_
    print('\nЛучшие параметры для', name, ':', search.best_params_)
    evaluate(name + ' tuned', search.best_estimator_)

#топ 10 признаков логистической регрессии
log_model = best['LogisticRegression'].named_steps['logisticregression']
coef = pd.Series(log_model.coef_[0], index=X.columns).abs().sort_values(ascending=False).head(10)
print('\nТоп-10 признаков LogisticRegression:\n', coef)
coef.sort_values().plot(kind='barh', title='Top-10 LogisticRegression features')
plt.tight_layout()
plt.show()

#дерево решений глубиной не больше 4
tree = best['DecisionTree']
plt.figure(figsize=(18, 8))
plot_tree(tree, feature_names=X.columns, class_names=['no disease','disease'],
          filled=True, max_depth=4, fontsize=8)
plt.title('Decision Tree, max depth = 4')
plt.tight_layout()
plt.show()
print('\nПризнак в корне дерева:', X.columns[tree.tree_.feature[0]])

#веса SVM с линейным ядром
svm_linear = make_pipeline(StandardScaler(), SVC(kernel='linear')).fit(Xtr, ytr)
svm_coef = pd.Series(svm_linear.named_steps['svc'].coef_[0], index=X.columns).abs().sort_values(ascending=False).head(10)
print('\nТоп-10 признаков SVM linear:\n', svm_coef)
