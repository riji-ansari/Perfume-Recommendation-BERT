# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# Note: If your environment doesn't support "!mamba install", use "!pip install"


#Keeping notes of a couple things:
#What I had to install: pip3 install -U sentence-transformers, pip3 install scikit-learn==0.23.1, pip3 install pandas, pip3 install seaborn, pip3 install matplotlib 

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


import skillsnetwork


#--------------------------------------------------- (importing data from SkillsNetwork)
import urllib.request
import tarfile

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX068IEN/data/perfume_data.tgz"
output_file = "perfume_data.tgz"

# Download the file
urllib.request.urlretrieve(url, output_file)

# Extract the contents of the tar file
with tarfile.open(output_file, "r:gz") as tar:
    tar.extractall()

# Remove the downloaded tar file
import os
os.remove(output_file)

#--------------------------------------------------- (importing data from SkillsNetwork)

sns.set_context('notebook')
sns.set_style('white')


def plotter(x, y, title):
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()


#SBERT for Sentence Embedding 
#SBERT has a Python library called sentence_transformers which you can use to compute sentence embeddings for more than 100 languages. These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. This can be useful for semantic textual similar, semantic search, or paraphrase mining.


sentences = ['This framework generates embeddings for each input sentence',
            'Sentences are passed as a list of string.',
            'The quick brown fox jumps over the lazy dog.']

model = SentenceTransformer('all-MiniLM-L6-v2')

#The .encode method of the model computes the embeddings of the sentences we pass in. The embeddings returned could be numpy arrays or tensors of length 384.
embeddings = model.encode(sentences, convert_to_numpy=True) # By default, convert_to_numpy = True
embeddings.shape

#We can examine the first 50 values in the embedding vector of the first sentence:
embeddings[0][:50]


#SBERT for Analyzing Semantic Textual Similarity (STS)
# We can calculate the similarity scores of the computed embeddings using e.g. cosine similarity to analyze the semantic relationship between sentences.

sentences = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

embeddings = model.encode(sentences, convert_to_numpy=True) 

# Let's define a function for calculating the cosine similarity score.

def cosine_similarity(a, b):
    
    score = np.dot(a, b) / (norm(a) * norm(b))
    
    return score

# The cosine_similarity score between "The cat sits outside" and "A man is playing guitar" should be low:

cosine_similarity(embeddings[0], embeddings[1])

# The cosine_similarity score between "The new movie is awesome" and "The new movie is so great" should be very high:

cosine_similarity(embeddings[3], embeddings[6])

# We can use the utility function cos_sim to calculate cosine similarity scores for all sentence pairs in sentences. Since we have 8 sentence embeddings, the function will return a  8×8 matrix of the scores.

cosine_scores = util.cos_sim(embeddings, embeddings)
cosine_scores.shape

# Let's create a list of the cosine scores of the unique sentence pairs. n sentence embeddings should generate $n(n-1) / 2$ pairs. Thus, 28 unique pairs in our case.

pairs = []

for i in range(len(cosine_scores)-1): # 0, 1, 2, 3, 4, 5, 6
    for j in range(i+1, len(cosine_scores)): # 1-7, 2-7, 3-7, 4-7, 5-7, 6-7, 7
        pairs.append({'index': [i,j], 'score': cosine_scores[i][j]})
        
len(pairs)

# We sort the scores in descending order and print 3 sentence pairs with the highest cosine similarity scores.

sorted_pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in sorted_pairs[0:3]:
    i, j = pair['index']
    print(f"{sentences[i]} | {sentences[j]} \n Score: {pair['score']:.2f} \n")

# We can also visualize the distance between the sentences on a 2-dimensional plot. Let's apply PCA to reduce the dimensionality of the sentence embeddings.

pca = PCA(n_components=2)
embeddings_reduced = pca.fit_transform(embeddings)

# By examining the plot below, we see that for the sentences that are semantically close to each other, they appear closely on the plot as well. For example 'The cat sits outside','The cat plays in the garden' are near and 'I love pasta' and 'Do you like pizza?' make up a cluster.

for coord, sentence in zip(embeddings_reduced, sentences):

    plt.scatter(coord[0], coord[1])
    plt.annotate(sentence, (coord[0], coord[1]))

# Use Case of SBERT Embeddings - Perfume Recommendation

# In this section, we will apply SBERT on a perfume dataset from Kaggle that contains the descriptions and notes of more than 2000 different types of perfumes. Since the notes of the perfumes are presented in natural language, SBERT will help us translate that into sentence embeddings which can be numerically manipulated.
# Preparing the Dataset

df = pd.read_csv("./perfume_data.csv", encoding="unicode_escape")
df.head()

# Let's examine the notes of the perfumes.

list(df.Notes[0:10])

# We want the text embeddings to be generated based on the Notes column. Thus, we remove redundant columns and organize the data frame.

df.rename(columns={"ï»¿Name": "Name"}, inplace=True)
df['Name'] = df['Brand'] + " - " + df['Name']
df.drop(labels=['Description', 'Image URL', 'Brand'], axis=1, inplace=True)
df.head()

# Check for missing values 

df.Notes.isnull().sum()

# Drop the rows with missing values and reset the indices.

df.dropna(inplace = True)
df.reset_index(inplace=True, drop = True)
df.shape

# We also want to drop other types of fragrances that are not perfumes (e.g. perfume oil, extract, travel spray, hair products, body spray, etc) but might have identical names and similar notes with the perfumes in the dataset.

words = ["Perfume Oil", "Extrait", "Travel", "Hair", "Body", "Hand", "Intense", "Intensivo", "Oil"] # check for these words in perfume names

index_to_drop = []
for index, name in enumerate(df.Name):
    if any(word.lower() in name.lower() for word in words):
        index_to_drop.append(index)

# After the preprocessing, we now have 1604 perfumes in the data frame.

df.drop(index_to_drop, axis=0, inplace=True)
df.reset_index(inplace=True, drop = True)
df.shape

# Creating Perfume Notes Embeddings
# Let's start by putting the 1604 perfume notes into a list.

df.Notes = df.Notes.apply(lambda x: str(x))
notes = df.Notes.to_list()
len(notes)

# We use the all-MiniLM-L6-v2 pre-trained SBERT model to generate the sentence embeddings of the perfume notes.

model = SentenceTransformer('all-MiniLM-L6-v2')

note_embeddings = model.encode(notes, show_progress_bar=True, batch_size=64)

# We check that each generated note embedding has length 384.

print(note_embeddings.shape)

print(note_embeddings[0][:50]) # first 50 values in the embedding of "Vanilla bean, musks"

# Recommending Perfumes using Cosine Similarity
# We calculate the cosine similarity scores for all the pairs of perfume note embeddings.

cosine_scores = util.cos_sim(note_embeddings, note_embeddings)
cosine_scores.shape

# Sorting the scores in descending order and appending the (index, score) pair to pairs.

pairs = []

for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({"index": [i,j], "score": cosine_scores[i][j]})

len(pairs)

# As we have more than 1000 perfumes in the dataset, the number of unique pairs of note embeddings is more than 1M. Next, we sort the pairs based on their corresponding similarity scores, and we print out 10 pairs with the highest scores as the most similar perfumes in the dataset.

sorted_pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in sorted_pairs[0:10]:
    i, j = pair['index']
    print(f"{df.iloc[i, 0]} | {df.iloc[j, 0]} \n Score: {pair['score']:.2f} \n")

# Exercise: Getting Your Own Perfume Suggestions
# Do you want to get recommendations for the perfume(s) you like or own? Add their names and notes (you can search it up here) to the my_perfumes dataframe!

my_perfumes = pd.DataFrame([['Jo Malone - English Pear & Freesia', 'Pear, Melon, Freesia, Rose, Musk, Patchouli, Rhuburb, Amber'], 
                      ['Jo Malone - Myrrh & Tonka', 'Lavender, Myrrh, Tonka Bean, Vanilla, Almond'],
                      ['Jo Malone - Oud & Bergamot', 'orange, bergamot, lemon, cedar and oud.'],
                      ['Guerlain - Néroli Outrenoir', 'Petitgrain, Bergamot, Tangerine, Lemon, Grapefruit, Tea, Neroli, Orange Blossom, Smoke, Earthy Notes, Myrrh, Vanilla, Benzoin, Ambrette, Oakmoss.'],
                      ['Guerlain - Épices Volées', 'Coriander, Lemon, Artemisia, Bergamot, Clove, Cardamom, Sage, Bulgarian Rose, Sandalwood, Patchouli, Benzoin, Labdanum.'],
                      ['Guerlain - Aqua Allegoria Nerolia Vetiver Eau de Toilette', 'Basil, Vetiver, Fig Accord, Neroli'],
                      ['Chloe Eau de Parfum', 'Peony, Litchi, Freesia, Rose, Lily-of-the-Valley, Magnolia, Virginia Cedar, Amber.']                     
                     ],
                   columns=df.columns)

my_perfumes

# Exercise 1: Create perfume embeddings
# Create perfume embeddings for my_perfumes using the all-MiniLM-L6-v2 model from sentence-transformer. Call it my_embeddings.
notes = list(my_perfumes.Notes)

model = SentenceTransformer('all-MiniLM-L6-v2')
my_embeddings = model.encode(notes, show_progress_bar=True)

# Exercise 2: Produce cosine similarity scores
# Calculate cosine similarity scores between my_perfumes and the other 1604 perfumes, i.e: a similarity matrix between my_embeddings and note_embeddings.
cosine_scores = util.cos_sim(my_embeddings, note_embeddings)

# Exercise 3: Sort the perfume similarity scores
# Create a list of (index, score) key-value pairs called my_pairs and sort my_pairs in descending order. Name the sorted list my_sorted_pairs.
my_pairs=[]

for i in range(cosine_scores.shape[0]):
    for j in range(cosine_score.shape[1]):
        my_pairs.append({"index": [i,j], "score": cosine_scores[i][j]})
        
        
my_sorted_pairs = sorted(my_pairs, key=lambda x: x['score'], reverse=True)

# For each of the perfume in my_perfumes, let's display the first 5 out of the 1604 perfumes that are most likely to be recommended.

for i in range(cosine_scores.shape[0]):

    print(f"Recommended for {my_perfumes.iloc[i, 0]}:")
    my_pairs = []
    for j in range(cosine_scores.shape[1]):
        my_pairs.append({"index": j, "score": cosine_scores[i][j]})
        my_sorted_pairs = sorted(my_pairs, key=lambda x: x['score'], reverse=True)
        
    for no, pair in enumerate(my_sorted_pairs[:5]):
        print(f" {no+1}. {df.iloc[pair['index'], 0]} (Score: {pair['score']:.2f})")
    print("\n")

# 
