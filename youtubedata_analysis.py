# Databricks notebook source
# Install necessary libraries
# %pip install google-api-python-client pandas

from googleapiclient.discovery import build
import pandas as pd

# Initialize the YouTube API client
DEVELOPER_KEY = "AIzaSyB52x8uxLfameKrof1MVW1qSlk77NCQV_A"
api_service_name = "youtube"
api_version = "v3"

youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Request to get comments for a specific video
video_id = "Ltnhz3YfJGY"
request = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    maxResults=100
)
response = request.execute()

# Extract comments into a list
comments = []
for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['updatedAt'],
        comment['likeCount'],
        comment['textDisplay']
    ])

# Convert the comments list to a Pandas DataFrame
pdf = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
pdf.head()


# COMMAND ----------

# MAGIC %md
# MAGIC Clean the Data

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql import functions as F


# COMMAND ----------

spark = SparkSession.builder \
    .appName("YouTube Data Cleaning") \
    .getOrCreate()

# Convert Pandas DataFrame to PySpark DataFrame
df = spark.createDataFrame(pdf)
df = df.dropna()


# COMMAND ----------

df = df.withColumn("like_count", col("like_count").cast("integer"))

# Clean the text column
df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z0-9\s]", ""))
df.show()

# COMMAND ----------

df = df.withColumn("text_cleaned", lower(regexp_replace(col("text"), "[^a-zA-Z0-9\s]", ""))) \
       .withColumn("text_cleaned", trim(col("text_cleaned")))

# Tokenization: Split text into words
tokenizer = Tokenizer(inputCol="text_cleaned", outputCol="words")
df = tokenizer.transform(df)

# Remove stop words (optional but recommended)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = remover.transform(df)

# Show the cleaned DataFrame
df.select("text_cleaned", "words", "filtered_words").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Perform Sentiment Analysis

# COMMAND ----------

# MAGIC %pip install textblob

# COMMAND ----------

from textblob import TextBlob
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType


# COMMAND ----------

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Register the UDF
sentiment_udf = udf(get_sentiment, FloatType())

# Add a new column with sentiment scores
df = df.withColumn("sentiment", sentiment_udf(df.text))
df.show()

# COMMAND ----------

# Summary statistics
df.describe().show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Convert PySpark DataFrame to Pandas for visualization
df_pd = df.select("sentiment").toPandas()

# Plot histogram of sentiment scores
plt.hist(df_pd['sentiment'], bins=20, alpha=0.75)
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.grid(True)
plt.show()


# COMMAND ----------

import plotly.express as px

# Example: Create a histogram of sentiment distribution
fig_hist = px.histogram(df.toPandas(), x="sentiment", title="Sentiment Distribution of YouTube Comments",
                        labels={'sentiment': 'Sentiment Polarity', 'count': 'Count'},
                        color_discrete_sequence=px.colors.qualitative.Set3)
fig_hist.show()

# Example: Create a line chart of sentiment trends over time
sentiment_over_time = df.groupBy("published_at").agg(F.avg("sentiment").alias("avg_sentiment")).toPandas()

fig_line = px.line(sentiment_over_time, x="published_at", y="avg_sentiment",
                   title="Sentiment Trends Over Time",
                   labels={'published_at': 'Date', 'avg_sentiment': 'Average Sentiment Polarity'},
                   color_discrete_sequence=px.colors.qualitative.Pastel)
fig_line.show()

# Example: Create a bar chart of average like counts by author
like_counts = df.groupBy("author").agg(F.avg("like_count").alias("avg_like_count")).toPandas()

fig_bar = px.bar(like_counts, x="author", y="avg_like_count",
                 title="Average Like Count by Author",
                 labels={'avg_like_count': 'Average Like Count', 'author': 'Author'},
                 color="author", color_discrete_sequence=px.colors.qualitative.Vivid)
fig_bar.show()

# Example: Create a scatter plot of sentiment vs like counts
fig_scatter = px.scatter(df.toPandas(), x='like_count', y='sentiment',
                         title='Sentiment vs Like Counts',
                         labels={'like_count': 'Like Count', 'sentiment': 'Sentiment Polarity'},
                         color='like_count', color_continuous_scale=px.colors.sequential.Viridis)
fig_scatter.show()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, CountVectorizer, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# COMMAND ----------



# COMMAND ----------

from pyspark.ml.feature import StringIndexer, CountVectorizer, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session (if not already initialized)
spark = SparkSession.builder \
    .appName("NaiveBayes Sentiment Analysis") \
    .getOrCreate()

# Assuming 'df' is your DataFrame with 'filtered_words' and 'sentiment' columns

# Drop existing 'label' column if it exists
if 'label' in df.columns:
    df = df.drop('label')

# Convert sentiment to numeric label using StringIndexer
indexer = StringIndexer(inputCol="sentiment", outputCol="label")
df = indexer.fit(df).transform(df)

# Split the data into training and test sets (80% training, 20% test)
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Vectorize the filtered words
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=1000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Create a Naive Bayes model
nb = NaiveBayes(featuresCol="features", labelCol="label")

# Build the pipeline
pipeline = Pipeline(stages=[vectorizer, idf, nb])

# Train the model
model = pipeline.fit(train)

# Make predictions
predictions = model.transform(test)

# Show prediction results
predictions.select("filtered_words", "sentiment", "prediction").show()

# Evaluate model performance
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Precision evaluation
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)
print(f"Precision: {precision}")



# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

# Train the Naive Bayes model
nb = NaiveBayes(featuresCol="features", labelCol="label")

# Build the pipeline
pipeline = Pipeline(stages=[vectorizer, idf, nb])

# Train the model
model = pipeline.fit(train)

# Extract the Naive Bayes model from the pipeline
nb_model = model.stages[-1]  # Assuming NaiveBayes is the last stage in your pipeline

# Specify the DBFS path to save the model
model_path = "dbfs:/ml/naive_bayes_sentiment_model"

# Save the Naive Bayes model
nb_model.write().overwrite().save(model_path)


# COMMAND ----------

print(os.listdir())

# COMMAND ----------

# MAGIC %fs ls dbfs:/ml/naive_bayes_sentiment_model
# MAGIC

# COMMAND ----------

cd /databricks/driver/metastore_db/tmp/

# COMMAND ----------

ls

# COMMAND ----------

import os

# Define the desired directory path
directory = '/dbfs/FileStore/tables/my_project_data/'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the CSV file to the specified directory
pdf.to_csv(directory + 'data.csv', index=False)


# COMMAND ----------


