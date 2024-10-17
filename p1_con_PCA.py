import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

misDatos = pd.read_csv('pokemons agrupados.csv')
# print(misDatos.columns[0])
# Eliminamos la columna indice y cluster

misDatos.drop(['Cluster'], axis=1, inplace=True)
misDatos.drop(misDatos.columns[0], axis=1, inplace=True)

# Encontramos la matriz de PCA
pca = PCA(n_components=10)
matriz_pca = pca.fit_transform(misDatos)


# Hacemos Kmedias
km = KMeans(n_clusters=17)
lista = km.fit_predict(matriz_pca)

cabeceras = ["PCA 1", "PCA 2", "PCA 3", "PCA 4", "PCA 5", "PCA 6", "PCA 7", "PCA 8", "PCA 9", "PCA 10"]

#Creamos un CSV con los PCA y los clusters
matrizPCA = pd.DataFrame(matriz_pca, columns=cabeceras)
matrizPCA['Cluster'] = lista
matrizPCA.to_csv('pokemons agrupados con PCA.csv')

