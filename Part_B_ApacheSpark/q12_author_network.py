from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import os
import re

spark = SparkSession.builder \
    .appName("AuthorInfluenceNetwork") \
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
            records.append((fname, f.read(5000)))

books_df = spark.createDataFrame(records, ["file_name", "text"])

MONTHS = (r"(?:January|February|March|April|May|June|July|"
          r"August|September|October|November|December)")

def extract_author_year(text):
    author_match = re.search(r"Author\s*:\s*(.+?)(?:\n|\r)", text, re.IGNORECASE)
    author = author_match.group(1).strip() if author_match else None
    
    date_match = re.search(
        rf"(?:Release\s+Date|Posting\s+Date)\s*:\s*{MONTHS}\s+\d{{1,2}},?\s+(\d{{4}})",
        text, re.IGNORECASE)
    year = int(date_match.group(1)) if date_match else None
    
    return (author, year)

author_udf = F.udf(
    extract_author_year,
    StructType([
        StructField("author", StringType()),
        StructField("year", IntegerType()),
    ])
)

author_df = books_df \
    .withColumn("info", author_udf("text")) \
    .select(
        "file_name",
        F.col("info.author").alias("author"),
        F.col("info.year").alias("year")) \
    .filter(F.col("author").isNotNull() & F.col("year").isNotNull())

author_df = author_df \
    .groupBy("author") \
    .agg(
        F.min("year").alias("year"),
        F.first("file_name").alias("file_name")) \
    .select("file_name", "author", "year")

X = 5

edges_df = author_df.alias("a") \
    .crossJoin(author_df.alias("b")) \
    .filter(F.col("a.author") != F.col("b.author")) \
    .filter(
        (F.col("b.year") >= F.col("a.year")) &
        (F.col("b.year") <= F.col("a.year") + X)
    ) \
    .select(
        F.col("a.author").alias("influencer"),
        F.col("b.author").alias("influenced")
    )

in_degree = edges_df \
    .groupBy("influenced") \
    .agg(F.count("influencer").alias("in_degree")) \
    .orderBy(F.col("in_degree").desc())

print("\nTop 5 authors with highest in-degree:")
in_degree.show(5, truncate=50)

out_degree = edges_df \
    .groupBy("influencer") \
    .agg(F.count("influenced").alias("out_degree")) \
    .orderBy(F.col("out_degree").desc())

print("\nTop 5 authors with highest out-degree:")
out_degree.show(5, truncate=50)

spark.stop()
