# Importando as bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregando o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de classificação (usando k-vizinhos mais próximos neste exemplo)
knn_model = KNeighborsClassifier(n_neighbors=3)

# Treinando o modelo
knn_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = knn_model.predict(X_test)

# Avaliando a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Exibindo o relatório de classificação
print('\nRelatório de Classificação:')
print(classification_report(y_test, y_pred))