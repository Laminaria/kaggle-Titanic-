import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import warnings

warnings.filterwarnings("ignore")
pd.options.display.max_columns = 15
pd.options.display.expand_frame_repr = False
df1 = pd.read_csv(r"C:\Users\User\Desktop\test.csv")
df2 = pd.read_csv(r"C:\Users\User\Desktop\train.csv")
#Женщины/Мужчины
male1 = df2[(df2['Sex'] == 'male') & (df2['Survived'] == 1)].shape[0]
male0 = df2[(df2['Sex'] == 'male') & (df2['Survived'] == 0)].shape[0]
female1 = df2[(df2['Sex'] == 'female') & (df2['Survived'] == 1)].shape[0]
female0 = df2[(df2['Sex'] == 'female') & (df2['Survived'] == 0)].shape[0]
mf1, mf0 = [male1, female1], [male0, female0]
MF = ['Мужчины', 'Женщины']
#Классы
class1_1 = df2[(df2['Pclass'] == 1) & (df2['Survived'] == 1)].shape[0]
class1_0 = df2[(df2['Pclass'] == 1) & (df2['Survived'] == 0)].shape[0]
class2_1 = df2[(df2['Pclass'] == 2) & (df2['Survived'] == 1)].shape[0]
class2_0 = df2[(df2['Pclass'] == 2) & (df2['Survived'] == 0)].shape[0]
class3_1 = df2[(df2['Pclass'] == 3) & (df2['Survived'] == 1)].shape[0]
class3_0 = df2[(df2['Pclass'] == 3) & (df2['Survived'] == 0)].shape[0]
classes_1, classes_0 = [class1_1, class2_1, class3_1], [class1_0, class2_0, class3_0]
clss = ['Класс1', 'Класс2', 'Класс3']
#По возрасту
age0_20_1 = df2[(df2['Age'] <= 20) & (df2['Survived'] == 1)].shape[0]
age0_20_0 = df2[(df2['Age'] <= 20) & (df2['Survived'] == 0)].shape[0]
age20_40_1 = df2[(df2['Age'] > 20) & (df2['Age'] <= 40) & (df2['Survived'] == 1)].shape[0]
age20_40_0 = df2[(df2['Age'] > 20) & (df2['Age'] <= 40) & (df2['Survived'] == 0)].shape[0]
age40_60_1 = df2[(df2['Age'] > 40) & (df2['Age'] <= 60) & (df2['Survived'] == 1)].shape[0]
age40_60_0 = df2[(df2['Age'] > 40) & (df2['Age'] <= 60) & (df2['Survived'] == 0)].shape[0]
age60_85_1 = df2[(df2['Age'] > 60) & (df2['Age'] <= 85) & (df2['Survived'] == 1)].shape[0]
age60_85_0 = df2[(df2['Age'] > 60) & (df2['Age'] <= 85) & (df2['Survived'] == 0)].shape[0]
age_1, age_0 = [age0_20_1, age20_40_1, age40_60_1, age60_85_1], [age0_20_0, age20_40_0, age40_60_0, age60_85_0]
age = ['(0;20]', '(20;40]', '(40;60]', '(60;85]']

fig = plt.figure(figsize=(7, 4))
fig.suptitle('Пассажиры Титаника')
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 1, 2)

x1 = np.arange(len(MF))
w = 0.3
ax1.bar(x1 - w/2, mf1, width=w, label='Выжили')
ax1.bar(x1 + w/2, mf0, width=w, label='Погибли')
ax1.set_xticks(x1)
ax1.set_xticklabels(MF)
ax1.set(ylim=(0, 500))
ax1.legend()
ax1.grid()

x2 = np.arange(len(clss))
ax2.bar(x2 - w/2, classes_1, width=w, label='Выжили')
ax2.bar(x2 + w/2, classes_0, width=w, label='Погибли')
ax2.set_xticks(x2)
ax2.set_xticklabels(clss)
ax2.legend()
ax2.grid()

x3 = np.arange(len(age))
ax3.bar(x3 - w/2, age_1, width=w, label='Выжили')
ax3.bar(x3 + w/2, age_0, width=w, label='Погибли')
ax3.set_xticks(x3)
ax3.set_xticklabels(age)
ax3.legend()
ax3.grid()
plt.show()

df = pd.concat([df1, df2])
df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
df.loc[df['Sex'] == 'male', 'Sex'], df.loc[df['Sex'] == 'female', 'Sex'] = 1, 0
df['Age'].fillna(df['Age'].median(), inplace=True)

x_test = df[df['Survived'].isna()]
x_test = x_test.drop(['Survived'], axis=1)

X = df[df['Survived'].notna()]
Y = X['Survived']
X = X.drop(['Survived'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13)
models = []
models.append(['KNN', KNeighborsClassifier()])
models.append(['LogReg', LogisticRegression(max_iter=10000)])
models.append(['SVM', LinearSVC(max_iter=10000)])
models.append(['Tree', DecisionTreeClassifier(max_depth=10000)])
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=43, shuffle=True)
    result = cross_val_score(model, X, Y, cv=kfold)
    print(f'{name}:{result.mean()}')
log = LogisticRegression()
log.fit(X_train, Y_train)
result = log.predict(x_test)
new_DF = pd.DataFrame({'PassengerId': x_test.PassengerId, 'Survived': result})
new_DF.Survived = new_DF.Survived.astype(int)
print(new_DF)

new_DF.to_csv(r'E:\projects\Numpy\TitanicPred.csv', index=False)
