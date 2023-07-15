# Perfume Recommender System

## Introduction

In our daily lives, we interact with people, and leaving a memorable impression on others can be challenging. Our sense of smell plays a significant role in creating memories. Selecting a personal perfume that not only evokes positive emotions but also reflects our unique identity can be a powerful way to make an impact. However, with the vast array of perfumes available, choosing the right one based on smell alone can be overwhelming. This project aims to address this challenge by leveraging machine learning to build a perfume recommender system based on the documented notes of perfumes.

[CognitiveClass.ai Project Tutorial:](https://cognitiveclass.ai/courses/course-v1:IBM+GPXX068IEN+v1)

## Project Description

This project focuses on designing a perfume recommendation system for a fragrance retailer. The goal is to create a system that suggests five perfumes to a customer based on the similarity of their notes to the customer's most recent perfume search. The project utilizes a dataset containing the notes of all the perfumes carried by the retailer from its primary perfumer.

## Code

The project code is implemented in Python and utilizes various libraries for data manipulation, embedding generation, and similarity calculations. Key libraries used in this project include:

- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SentenceTransformers

The code performs the following steps:

1. Imports the necessary libraries
2. Downloads and extracts the perfume dataset from SkillsNetwork
3. Loads and preprocesses the dataset
4. Generates sentence embeddings using SBERT (Sentence-BERT)
5. Calculates cosine similarity scores between perfume embeddings
6. Sorts the similarity scores and identifies the most similar perfumes
7. Recommends perfumes based on the most similar ones to the user's input

## Usage

To utilize the perfume recommender system, follow these steps:

1. Install the required libraries by running the provided commands.
2. Download the perfume dataset using the provided code.
3. Run the main code to generate perfume embeddings and calculate similarity scores.
4. Customize the `my_perfumes` dataframe with the perfumes you like or own, specifying their names and notes.
5. Run the code to receive perfume recommendations based on your input.

## Results

The perfume recommender system employs SBERT embeddings and cosine similarity to identify perfumes with similar notes. Based on the provided dataset, the system recommends five perfumes that are highly likely to be similar to the user's input.

## Conclusion

The perfume recommender system offers a convenient solution for customers to explore and discover new perfumes based on their preferences. By harnessing the power of machine learning, the system analyzes perfume notes and provides recommendations for similar perfumes, enhancing the customer experience and potentially boosting online sales for the fragrance retailer.

---

Feel free to include this rewritten description in your project repository, making any further adjustments or additions you deem necessary.
