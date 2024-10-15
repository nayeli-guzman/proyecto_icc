import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --------- loading stopwords

stopWords = pd.read_csv("stopwords.csv")
stopWords = stopWords.iloc[:, 0].tolist()

# --------- reading data

data = pd.read_csv("smogon.csv")
# print(data)

# --------- generating tfidf matrix

vector = TfidfVectorizer(stop_words=stopWords, ngram_range=(1, 3 ))
x = vector.fit_transform(data["moves"])
frequencyMatrix = pd.DataFrame(data=x.toarray(), columns=sorted(vector.vocabulary_))

# --------- tokens

print("El vocabulario es:")
# print(sorted(vector.vocabulary_))

print("La cantidad de tokens es:")
print(len(sorted(vector.vocabulary_)))


# --------- df tfidf matrix

print("Matriz de frecuencias:")
# print(frequencyMatrix)

# ----------- using kmeans

km = KMeans(n_clusters=18, n_init=100)
clustersList = km.fit_predict(frequencyMatrix)
print(clustersList)
frequencyMatrix["Cluster"] = clustersList

frequencyMatrix.to_csv("pokemons agrupados")