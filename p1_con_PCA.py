import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Cargamos en un DataFrame el archivo CSV generado en p1
misDatos = pd.read_csv('pokemonesAgrupados1.csv')
data = pd.read_csv("smogon.csv")
pokemones_clusters_sinPCA = pd.read_csv("pokemonesClusters1.csv")

# Eliminamos la columna indice y cluster
misDatos.drop(misDatos.columns[0], axis=1, inplace=True)
misDatos.drop(['Cluster'], axis=1, inplace=True)
print(misDatos)

# Elaboramos la matriz de PCA
pca = PCA(n_components=12)
matriz_pca = pca.fit_transform(misDatos)
print("Filas y columnas del dataframe original: ",misDatos.shape)
print("Filas y columnas de la matriz de PCA: ",matriz_pca.shape)

# Generacion del nuevo Dataframe
cabeceras = ["PCA 1", "PCA 2", "PCA 3", "PCA 4", "PCA 5", "PCA 6", "PCA 7", "PCA 8", "PCA 9", "PCA 10"]
matrizPCA = pd.DataFrame(matriz_pca, columns=cabeceras)

# Empleamos Kmedias para agrupar las filas
km = KMeans(n_clusters=18)
clusters_con_PCA = km.fit_predict(matriz_pca)


#Creamos un CSV con los PCA y los clusters
matrizPCA['Cluster'] = clusters_con_PCA
matrizPCA.to_csv('pokemons agrupados con PCA.csv')

table = pd.DataFrame({
    "Pokemon": data["Pokemon"],
    "Cluster": clusters_con_PCA
})
table.to_csv("pokemonesClusters3.csv")

from sklearn.metrics import adjusted_rand_score

# Comparación usando ARI
ari_score = adjusted_rand_score(pokemones_clusters_sinPCA["Cluster"], clusters_con_PCA)
print(f"ARI Score: {ari_score}")

