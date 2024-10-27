import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Carregando o dataset Heart Disease da UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv(url, names=columns)

# Substituindo valores "?" por NaN e removendo as linhas com valores faltantes
df = df.replace("?", pd.NA).dropna()

# Transformando colunas para tipos numéricos
df = df.astype(float)

# Criando uma variável binária de diagnóstico de doença cardíaca
df['num'] = (df['num'] > 0).astype(int)

# Dividindo as features (X) e o alvo (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializando o SGDClassifier
sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Treinando o modelo
sgd.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = sgd.predict(X_test)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

# Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Curva ROC
y_pred_proba = sgd.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"\nAUC-ROC: {roc_auc:.2f}")

# Plotando gráficos

# 1. Distribuição de Idade por Diagnóstico
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', hue='num', multiple='stack', bins=20, palette='Set1')
plt.title('Distribuição de Idade por Diagnóstico de Doença Cardíaca')
plt.xlabel('Idade')
plt.ylabel('Contagem')
plt.show()

# 2. Matriz de Confusão
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Valor Predito')
plt.ylabel('Valor Verdadeiro')
plt.show()

# 3. Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# 4. Relação entre Nível de Colesterol e Doença Cardíaca
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='chol', y='age', hue='num', palette='Set1')
plt.title('Relação entre Nível de Colesterol e Idade por Diagnóstico')
plt.xlabel('Colesterol')
plt.ylabel('Idade')
plt.show()

