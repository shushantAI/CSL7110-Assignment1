from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import os
import re

spark = SparkSession.builder \
    .appName("BookTFIDF") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

BOOKS_DIR = os.path.expanduser("~/Downloads/D184MB")
records = []
for fname in os.listdir(BOOKS_DIR):
    if fname.endswith(".txt"):
        with open(os.path.join(BOOKS_DIR, fname),
                  encoding="utf-8", errors="ignore") as f:
            records.append((fname, f.read(50000)))

books_df = spark.createDataFrame(records, ["file_name", "text"])

HEADER_RE = re.compile(r"\*{3}.*?START.*?\*{3}", re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*{3}.*?END.*", re.IGNORECASE | re.DOTALL)

STOP_WORDS = set(["the","a","an","and","or","but","in","on","at","to",
    "for","of","with","is","was","are","were","be","been","has","have",
    "had","it","its","this","that","i","he","she","they","we","you",
    "not","by","from","as","so","if","do","did","will","would","could",
    "should","may","might","which","who","what","when","where","how"])

def clean_text(text):
    text = HEADER_RE.sub("", text)
    text = FOOTER_RE.sub("", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return tokens

clean_udf = F.udf(clean_text, "array<string>")
words_df = books_df.withColumn("words", clean_udf("text")).select("file_name", "words")

tf_raw = words_df \
    .select("file_name", F.explode("words").alias("word")) \
    .groupBy("file_name", "word") \
    .agg(F.count("*").alias("word_count"))

total_words = words_df.select("file_name", F.size("words").alias("total_words"))

tf_df = tf_raw.join(total_words, "file_name") \
    .withColumn("TF", F.col("word_count") / F.col("total_words"))

num_docs = books_df.count()
df_count = tf_raw.groupBy("word").agg(F.countDistinct("file_name").alias("doc_freq"))
idf_df = df_count.withColumn("IDF", F.log(F.lit(num_docs + 1) / (F.col("doc_freq") + 1)))

tfidf_df = tf_df.join(idf_df, "word") \
    .withColumn("tfidf", F.col("TF") * F.col("IDF")) \
    .select("file_name", "word", "tfidf")

tfidf_df.write.mode("overwrite").parquet("/tmp/tfidf_cache")
tfidf_df = spark.read.parquet("/tmp/tfidf_cache")

print("\nTop TF-IDF words sample:")
tfidf_df.orderBy(F.col("tfidf").desc()).show(10)

from pyspark.sql import Window
w = Window.partitionBy("file_name").orderBy(F.col("tfidf").desc())
top_df = tfidf_df \
    .withColumn("rank", F.rank().over(w)) \
    .filter(F.col("rank") <= 100) \
    .select("file_name", "word", "tfidf")

norms = top_df.groupBy("file_name") \
    .agg(F.sqrt(F.sum(F.col("tfidf") ** 2)).alias("norm"))

pair_df = top_df.alias("a") \
    .join(top_df.alias("b"), "word") \
    .filter(F.col("a.file_name") < F.col("b.file_name")) \
    .select(
        F.col("a.file_name").alias("book1"),
        F.col("b.file_name").alias("book2"),
        (F.col("a.tfidf") * F.col("b.tfidf")).alias("dot")) \
    .groupBy("book1", "book2") \
    .agg(F.sum("dot").alias("dot_product"))

sim_df = pair_df \
    .join(norms.alias("n1"), F.col("book1") == F.col("n1.file_name")) \
    .join(norms.alias("n2"), F.col("book2") == F.col("n2.file_name")) \
    .withColumn("cosine_sim", F.col("dot_product") / (F.col("n1.norm") * F.col("n2.norm"))) \
    .select("book1", "book2", "cosine_sim")

sim_df.write.mode("overwrite").parquet("/tmp/sim_cache")
sim_df = spark.read.parquet("/tmp/sim_cache")

query = "10.txt"
print(f"\nTop 5 books similar to {query}:")
sim_df.filter((F.col("book1") == query) | (F.col("book2") == query)) \
    .select(
        F.when(F.col("book1") == query, F.col("book2"))
         .otherwise(F.col("book1")).alias("similar_book"),
        "cosine_sim") \
    .orderBy(F.col("cosine_sim").desc()) \
    .show(5)

spark.stop()
