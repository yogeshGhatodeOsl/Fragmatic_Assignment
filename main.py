#!/usr/bin/env python

import click
import pandas as pd
import time
from pymongo import MongoClient

import nltk
from nltk.corpus import stopwords

import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')
nlp = spacy.load('en_core_web_sm')

client = MongoClient('localhost', 27017) 
db = client['fragmatic_assignment']
collection = db['dataset_all']


@click.group()
def cli():
    pass

@cli.command()
@click.argument('csv_path')
def import_headlines(csv_path):
    start_time = time.time()
    
    try:
        # loading the csv data
        df = pd.read_csv(csv_path)
        
        #nlp processing
        nltk.download('stopwords')

        stop_words = set(stopwords.words('english'))
        df['headline_text'] = df['headline_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

        data = df.to_dict(orient='records')

        #storing in database
        collection.insert_many(data)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Data ingestion completed in {execution_time:.2f} seconds.")

    except Exception as e:
        print(f"Error: {str(e)}")



#  second task :

@cli.command()
def extract_entities_sentiment():
    start_time = time.time()

    cursor = collection.find()

    for document in cursor:
        text = document["headline_text"] 
        doc = nlp(text)
        # entities = [(ent.text, ent.label_) for ent in doc.ents]
        entities = [{ "ent_name" : ent.label_, "ent_text" : ent.text } for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "LOC"]]
        sentiment = sid.polarity_scores(text)
        sentiment_label = "positive" if sentiment["compound"] > 0 else "negative" if sentiment["compound"] < 0 else "neutral"
        
        # Update the document in the collection
        collection.update_one({"_id": document["_id"]},{"$set": {"entities": entities, "sentiment": sentiment_label}})


    print("Attributes added to all documents in the collection.")

    end_time = time.time()
    execution_time = end_time - start_time
    
    print("Execution time:", execution_time, "seconds")



# third task : 

from collections import Counter

@cli.command()
def retrieve_top_100_entities_with_type():
    start_time = time.time()

    documents = collection.find({}, {"entities": 1})

    entity_counter = Counter()
    # entity_name_arr = []
    
    allowed_entity_types = ["PERSON", "ORG", "LOC"]

    for document in documents:
        entities = document.get("entities", [])
        for entity in entities:
            
            entity_text = entity.get("ent_text")
            entity_type = entity.get("ent_name")

            if entity_type in allowed_entity_types:
                entity_counter[(entity_text, entity_type)] += 1

    
    top_100_entities = entity_counter.most_common(100)
    

    for i, (entity, count) in enumerate(top_100_entities, start=1):
        entity_text, entity_type = entity
        # print(f"{i}. Entity Text: {entity_text}, Entity Name: {entity_type}, Frequency: {count}")
        print(f"Entity Text: {entity_text}")
        # entity_name_arr.append(entity_text)   # adding all entity name came in top_100_entities.


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")
    # return entity_name_arr


# task 4 :
@cli.command()
@click.argument('entity_name')
def retrieve_all_headlines_for_entity(entity_name):
    start_time = time.time()

    cursor = collection.find({"entities.ent_text": entity_name}, {"headline_text": 1})


    # print(f"Headlines associated with entity '{entity_name}':")
    for i, document in enumerate(cursor, start=1):
        headline_text = document.get("headline_text", "")
        print(headline_text)

    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution Time: {execution_time} seconds")



cli.add_command(import_headlines)
cli.add_command(extract_entities_sentiment)
cli.add_command(retrieve_top_100_entities_with_type)
cli.add_command(retrieve_all_headlines_for_entity)

if __name__ == "__main__":
    cli()