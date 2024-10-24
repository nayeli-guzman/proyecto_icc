import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --------- loading stopwords

stopWords = pd.read_csv("stopwords.csv")
stopWords = stopWords.iloc[:, 0].tolist()

# --------- reading data

data = pd.read_csv("smogon.csv")
data.drop(["url"], axis=1, inplace=True)

# --------- types

types = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", 
    "Fighting", "Poison", "Ground", "Flying", "Psychic", 
    "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", 
    "Fairy"
]

data["moves"] = data["moves"].str.extractall("(" + "|".join(types) + ")")[0].groupby(level=0).apply(" ".join)
data["moves"].fillna("", inplace=True) 
print(data)

# --------- generating tfidf matrix

vector = TfidfVectorizer(stop_words=stopWords, ngram_range=(1,1))
x = vector.fit_transform(data["moves"])
frequencyMatrix = pd.DataFrame(data=x.toarray(), columns=sorted(vector.vocabulary_))

# --------- tokens

print("Los tokens son:")
print(sorted(vector.vocabulary_))

print("La cantidad de tokens es:")
print(len(sorted(vector.vocabulary_)))

# ----------- using kmeans

km = KMeans(n_clusters=18, n_init=10)
clustersList = km.fit_predict(frequencyMatrix)
print("Clustering:")
print(clustersList)
frequencyMatrix["Cluster"] = clustersList

frequencyMatrix.to_csv("pokemonesAgrupados2.csv")

table = pd.DataFrame({
    "Pokemon": data["Pokemon"], 
    "Cluster": clustersList
})
table.to_csv("pokemonesClusters2.csv")
