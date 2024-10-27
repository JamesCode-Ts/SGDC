# Análise e Classificação de Doença Cardíaca com Machine Learning

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

*Carregando o dataset Heart Disease da UCI
# Utilizamos uma URL pública que contém dados sobre doenças cardíacas.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv(url, names=columns)

# Substituindo valores "?" por NaN e removendo as linhas com valores faltantes
# Isso é necessário pois valores faltantes podem prejudicar a qualidade dos dados e do modelo.
df = df.replace("?", pd.NA).dropna()

# Transformando colunas para tipos numéricos
# Essa conversão facilita operações numéricas e o processamento pelo modelo de ML.
df = df.astype(float)

# Criando uma variável binária de diagnóstico de doença cardíaca
# Convertendo a coluna "num" para 0 e 1, onde 1 indica presença de doença cardíaca.
df['num'] = (df['num'] > 0).astype(int)

# Dividindo as features (X) e o alvo (y)
# O dataset é dividido entre variáveis de entrada (X) e a saída (y), que é o diagnóstico.
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Dividindo o dataset em treino e teste
# 70% dos dados para treino e 30% para teste, garantindo aleatoriedade com um estado fixo.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizando os dados
# A padronização melhora a performance do modelo ao reduzir a variância das features.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializando o SGDClassifier
# Este é o classificador linear baseado em gradiente descendente estocástico, ajustado para 1000 iterações.
sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Treinando o modelo
# Treinamento realizado no conjunto de dados de treino.
sgd.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
# Utilizamos o modelo treinado para prever os valores de y com base em X_test.
y_pred = sgd.predict(X_test)

# Avaliando a acurácia
# Acurácia mede a precisão do modelo em prever corretamente o diagnóstico.
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")

# Matriz de Confusão
# Exibe uma tabela de acertos e erros, útil para entender onde o modelo acerta e erra.
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

# Relatório de Classificação
# Inclui precisão, recall e F1-score para cada classe.
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Curva ROC
# Mede a performance do modelo em classificar verdadeiros positivos e falsos positivos.
y_pred_proba = sgd.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"\nAUC-ROC: {roc_auc:.2f}")

# Plotando gráficos

# 1. Distribuição de Idade por Diagnóstico
# Visualiza como a idade dos pacientes se distribui entre diagnosticados e não diagnosticados com doença cardíaca.
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', hue='num', multiple='stack', bins=20, palette='Set1')
plt.title('Distribuição de Idade por Diagnóstico de Doença Cardíaca')
plt.xlabel('Idade')
plt.ylabel('Contagem')
plt.show()

# 2. Matriz de Confusão
# Gráfico de mapa de calor mostrando a matriz de confusão.
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Valor Predito')
plt.ylabel('Valor Verdadeiro')
plt.show()

# 3. Curva ROC
# Representação gráfica da taxa de verdadeiros positivos versus falsos positivos.
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
# Analisa a relação entre colesterol e idade em relação ao diagnóstico.
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='chol', y='age', hue='num', palette='Set1')
plt.title('Relação entre Nível de Colesterol e Idade por Diagnóstico')
plt.xlabel('Colesterol')
plt.ylabel('Idade')
plt.show()
