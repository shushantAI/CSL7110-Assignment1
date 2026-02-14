from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
import os
import re

spark = SparkSession.builder \
    .appName("BookMetadata") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

BOOKS_DIR = os.path.expanduser("~/Downloads/D184MB")

records = []
for fname in os.listdir(BOOKS_DIR):
    if fname.endswith(".txt"):
        fpath = os.path.join(BOOKS_DIR, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(5000)
            records.append((fname, content))

books_df = spark.createDataFrame(records, ["file_name", "text"])
print(f"Total books loaded: {books_df.count()}")

def extract_metadata(text):
    title_m = re.search(r"Title\s*:\s*(.+)", text, re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else None
    date_m = re.search(r"(?:Release\s+Date|Posting\s+Date)\s*:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})", text, re.IGNORECASE)
    release_date = date_m.group(1).strip() if date_m else None
    lang_m = re.search(r"Language\s*:\s*(\S+)", text, re.IGNORECASE)
    language = lang_m.group(1).strip() if lang_m else None
    enc_m = re.search(r"(?:Character\s+set\s+encoding|Encoding)\s*:\s*(\S+)", text, re.IGNORECASE)
    encoding = enc_m.group(1).strip() if enc_m else None
    return (title, release_date, language, encoding)

extract_udf = F.udf(extract_metadata, StructType([
    StructField("title", StringType()),
    StructField("release_date", StringType()),
    StructField("language", StringType()),
    StructField("encoding", StringType())
]))

meta_df = books_df \
    .withColumn("meta", extract_udf("text")) \
    .select("file_name", "meta.title", "meta.release_date", "meta.language", "meta.encoding")

meta_df.write.mode("overwrite").parquet("/tmp/meta_cache")
meta_df = spark.read.parquet("/tmp/meta_cache")

print("\nSample Metadata:")
meta_df.show(5, truncate=60)

year_udf = F.udf(lambda rd: re.search(r"\d{4}", rd).group() if rd and re.search(r"\d{4}", rd) else None, StringType())

print("\nBooks released per year (top 5):")
meta_df.withColumn("year", year_udf("release_date")) \
    .filter(F.col("year").isNotNull()) \
    .groupBy("year").count() \
    .orderBy(F.col("count").desc()) \
    .show(5)

print("\nMost common language:")
meta_df.filter(F.col("language").isNotNull()) \
    .groupBy("language").count() \
    .orderBy(F.col("count").desc()) \
    .show(5)

print("\nAverage title length:")
meta_df.filter(F.col("title").isNotNull()) \
    .select(F.avg(F.length("title")).alias("avg_title_length")) \
    .show()

spark.stop()
