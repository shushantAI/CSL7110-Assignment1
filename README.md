# CSL7110 – Assignment 1 (Hadoop MapReduce + Apache Spark)

**Name:** Shushant Kumar Tiwari  
**Roll Number:** M25DE1071  

This repository contains my solutions for **CSL7110 Assignment 1**, split into two parts:
- **Part A:** Apache Hadoop / MapReduce (Java)
- **Part B:** Apache Spark (PySpark)

The assignment focuses on running the classic **WordCount** job on Hadoop and then doing text analytics on the Project Gutenberg books dataset using Spark (metadata extraction + TF-IDF + similarity + a simple influence network).

---

## Repo Structure

- `Part_A_ApacheHadoop/` → MapReduce (Java) work for WordCount / Hadoop-related questions  
- `Part_B_ApacheSpark/` → PySpark work (DataFrame/RDD based analysis)
  
---

## Requirements

### For Hadoop / MapReduce (Part A)
- Java 8+
- Apache Hadoop (single-node / pseudo-distributed setup)
- Ability to run HDFS commands + Hadoop jobs

### For Spark (Part B)
- Java 8+
- Apache Spark
- Python 3.11 + PySpark

---

## Dataset Used

The assignment uses the **Project Gutenberg books dataset**.  
In my runs, I loaded it into Spark as a DataFrame with columns like:
- `file_name` (string)
- `text` (string)

---

## Part A: Apache Hadoop

### What I did
- Verified Hadoop + HDFS setup by running the standard **WordCount** example
- Implemented/edited `WordCount.java` using Hadoop `Writable` types
- Ran WordCount on sample input and on the required `200.txt` file

### Typical run flow 
> Commands may vary slightly depending on local installation paths.

1. Start services
   - `start-dfs.sh`
   - `start-yarn.sh`

2. Copy input to HDFS and run the job
   - `hdfs dfs -mkdir -p /user/<user>/wordcount/input`
   - `hdfs dfs -put <input_file> /user/<user>/wordcount/input/`
   - compile + jar
   - run jar on Hadoop
   - `hdfs dfs -cat /user/<user>/wordcount/output/part-r-00000`

---

## Part B: Apache Spark (PySpark)

### What I did
1. **Metadata extraction** from raw Gutenberg text using regex:
   - title
   - release date
   - language
   - encoding
2. **Analysis**
   - books released per year
   - most common language
   - average title length
3. **TF-IDF + cosine similarity**
   - cleaned text (lowercase, punctuation removal, tokenization, stopword removal)
   - computed TF-IDF vectors
   - computed cosine similarity and returned top-k similar books (example: for `"10.txt"`)
4. **Author influence network (simplified)**
   - author + release year extraction
   - created edges if release years are within X years (like X=5)
   - computed in-degree / out-degree and top authors

---

## How to Use This Repo
- Open `Part_A_ApacheHadoop/` for Hadoop/Java files and run them on a Hadoop setup.
- Open `Part_B_ApacheSpark/` for PySpark scripts/notebooks and run them with Spark.

---
