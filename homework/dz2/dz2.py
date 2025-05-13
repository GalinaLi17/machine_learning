import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('AmesHousing.csv')

#пропуски: None-объекта нет;0-числовое знач
none_cols = ['Alley','Pool QC','Fence','Misc Feature','Fireplace Qu','Garage Type','Garage Finish','Garage Qual','Garage Cond','Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','BsmtFin Type 2']
zero_cols = ['Bsmt Full Bath','Bsmt Half Bath','Garage Cars','Garage Area','Total Bsmt SF','BsmtFin SF 1','BsmtFin SF 2','Bsmt Unf SF']
for c in none_cols:
    if c in df: df[c] = df[c].fillna('None')
for c in zero_cols:
    if c in df: df[c] = df[c].fillna(0)
df['Lot Frontage'] = df['Lot Frontage'].fillna(df.groupby('Neighborhood')['Lot Frontage'].transform('median'))
for c in df.select_dtypes('number'): df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes('object'): df[c] = df[c].fillna(df[c].mode()[0])

#сезоны + доп возраст дома, лет с ремонта
seasons = {12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'}
df['House Age'] = df['Yr Sold'] - df['Year Built']
df['Years Since Remodel'] = df['Yr Sold'] - df['Year Remod/Add']
df['Season'] = df['Mo Sold'].map(seasons)

#Ridge
X = pd.get_dummies(df.drop(columns=['SalePrice','Order','PID']), drop_first=True)
y = df['SalePrice']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(Xtr, ytr)
pred = model.predict(Xte)
print('Ridge до аномалий: R2 =', round(r2_score(yte, pred), 3), 'MAE =', round(mean_absolute_error(yte, pred)))

#10 признаков по модулю коэффициента
coef = pd.Series(model.named_steps['ridge'].coef_, index=X.columns).abs().sort_values(ascending=False).head(10)
print('\nТоп-10 важных признаков Ridge:\n', coef)
coef.sort_values().plot(kind='barh', title='Top-10 Ridge features'); plt.tight_layout(); plt.show()

#график и межквартальный разброс
plt.scatter(df['Gr Liv Area'], df['SalePrice'], alpha=0.5)
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.title('Цена от жилой площади')
plt.tight_layout()
plt.show()

price_per_area = df['SalePrice'] / df['Gr Liv Area']
Q1 = price_per_area.quantile(0.25)
Q3 = price_per_area.quantile(0.75)
IQR = Q3 - Q1
low_price = price_per_area < Q1 - 1.5 * IQR
big_area = df['Gr Liv Area'] > df['Gr Liv Area'].quantile(0.75)
anomaly = low_price & big_area
print('\nАномальных дешевых больших домов:', anomaly.sum())

#сравнение качества после удаления аномалий
X_clean = pd.get_dummies(df[~anomaly].drop(columns=['SalePrice','Order','PID']), drop_first=True).reindex(columns=X.columns, fill_value=0)
y_clean = df.loc[~anomaly, 'SalePrice']
Xtr, Xte, ytr, yte = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
model.fit(Xtr, ytr); pred = model.predict(Xte)
print('Ridge после аномалий: R2 =', round(r2_score(yte, pred), 3), 'MAE =', round(mean_absolute_error(yte, pred)))

#сегменты недвижимости без учета цены
seg_cols = ['Gr Liv Area','Lot Area','Overall Qual','Overall Cond','House Age','Garage Area']
df['Segment'] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(df[seg_cols]))
print('\nСегменты:\n', df.groupby('Segment')[seg_cols].mean().round(1))

#PCA по числовым признакам + регрессия
num = df.select_dtypes('number').drop(columns=['SalePrice','Order','PID','Segment'])
X_pca = make_pipeline(StandardScaler(), PCA(n_components=10)).fit_transform(num)
Xtr, Xte, ytr, yte = train_test_split(X_pca, y, test_size=0.2, random_state=42)
pred = Ridge(alpha=10).fit(Xtr, ytr).predict(Xte)
print('\nRidge на PCA: R2 =', round(r2_score(yte, pred), 3), 'MAE =', round(mean_absolute_error(yte, pred)))

#кризис 2008 и сезонности
print('\nСредняя цена по годам:\n', df.groupby('Yr Sold')['SalePrice'].mean().round())
print('\nСредняя цена по месяцам:\n', df.groupby('Mo Sold')['SalePrice'].mean().round())
print('\nСредняя цена по сезонам:\n', df.groupby('Season')['SalePrice'].mean().round())
change_2008 = (df[df['Yr Sold']==2008]['SalePrice'].mean() / df[df['Yr Sold']==2007]['SalePrice'].mean() -1 ) * 100
print('\nИзменение цены в 2008 относительно 2007:', round(change_2008, 2), '%')
df.groupby('Yr Sold')['SalePrice'].mean().plot(marker='o', title='Динамика цен по годам'); plt.ylabel('SalePrice'); plt.tight_layout(); plt.show()
df.groupby('Mo Sold')['SalePrice'].mean().plot(marker='o', title='Сезонность по месяцам'); plt.ylabel('SalePrice'); plt.tight_layout(); plt.show()
