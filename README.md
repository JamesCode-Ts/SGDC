
# Análise e Classificação de Doença Cardíaca

Este documento descreve um projeto que utiliza aprendizado de máquina para classificar a presença de doenças cardíacas com base em um conjunto de dados.

## Importando Bibliotecas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
```

Esse bloco importa as bibliotecas necessárias para manipulação de dados (pandas), cálculos matemáticos (numpy), visualização (matplotlib e seaborn) e criação do modelo de classificação (scikit-learn).

## Carregando o Dataset Heart Disease

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv(url, names=columns)
```

Aqui, o código carrega o conjunto de dados de doenças cardíacas e define os nomes das colunas para facilitar o uso.

## Tratamento de Dados Faltantes

```python
df = df.replace("?", pd.NA).dropna()
```

Os valores "?" são substituídos por NaN (indicando valores ausentes), e as linhas com valores faltantes são removidas para limpar os dados.

## Conversão de Tipos de Dados

```python
df = df.astype(float)
```

Converte todas as colunas para o tipo float, facilitando o processamento numérico.

## Criando uma Variável Binária para Diagnóstico

```python
df['num'] = (df['num'] > 0).astype(int)
```

A coluna `num` é transformada em uma variável binária, onde 1 indica a presença de doença cardíaca e 0 indica ausência.

## Dividindo as Features e o Alvo

```python
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

Aqui, o conjunto de dados é dividido entre features (`X`) e o alvo (`y`), sendo `y` o diagnóstico.

## Dividindo o Dataset em Treinamento e Teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

O conjunto de dados é dividido entre treino e teste, com 70% dos dados para treino e 30% para teste.

## Padronizando os Dados

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Os dados são padronizados para melhorar a performance do modelo.

## Inicializando e Treinando o SGDClassifier

```python
sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)
```

Inicializa o classificador `SGDClassifier` e treina o modelo com o conjunto de treino.

## Fazendo Previsões no Conjunto de Teste

```python
y_pred = sgd.predict(X_test)
```

O modelo faz previsões no conjunto de teste.

## Avaliação do Modelo

### Acurácia

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")
```

Calcula a acurácia do modelo, que indica a porcentagem de previsões corretas.

### Matriz de Confusão

```python
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)
```

Gera a matriz de confusão, que mostra os acertos e erros em cada classe.

### Relatório de Classificação

```python
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
```

Gera um relatório de classificação que inclui métricas como precisão, recall e F1-score para cada classe.

### Curva ROC e AUC

```python
y_pred_proba = sgd.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"\nAUC-ROC: {roc_auc:.2f}")
```

Calcula a Curva ROC e a AUC, que medem a capacidade de discriminação do modelo.

## Visualizações

### Distribuição de Idade por Diagnóstico

```python
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', hue='num', multiple='stack', bins=20, palette='Set1')
plt.title('Distribuição de Idade por Diagnóstico de Doença Cardíaca')
plt.xlabel('Idade')
plt.ylabel('Contagem')
plt.show()
```

Mostra a distribuição de idade para pacientes com e sem doença cardíaca.

### Matriz de Confusão

```python
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Valor Predito')
plt.ylabel('Valor Verdadeiro')
plt.show()
```

Visualiza a matriz de confusão com um mapa de calor.

### Curva ROC

```python
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
```

Exibe a Curva ROC para avaliar o desempenho do modelo.

### Relação entre Nível de Colesterol e Doença Cardíaca

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='chol', y='age', hue='num', palette='Set1')
plt.title('Relação entre Nível de Colesterol e Idade por Diagnóstico')
plt.xlabel('Colesterol')
plt.ylabel('Idade')
plt.show()
```

Mostra a relação entre níveis de colesterol e idade em relação ao diagnóstico de doença cardíaca.

## Conclusão

Este projeto ilustra o uso de aprendizado de máquina para classificar a presença de doenças cardíacas com base em um conjunto de dados e fornece visualizações para melhor compreensão dos resultados.
