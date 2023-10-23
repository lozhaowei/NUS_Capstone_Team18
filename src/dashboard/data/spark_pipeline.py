import sys
import os
import time
from pyspark.sql import SparkSession
from decouple import config

class SparkPipeline:
    url = f"jdbc:mysql://{config('DB_HOST')}:{config('DB_PORT')}/{config('DB_NAME')}?useSSL=false"
    spark: SparkSession

    def __init__(self):
        # Initialize HADOOP_HOME environment variable for spark
        os.environ['HADOOP_HOME'] = "C:/hadoop-3.3.6"
        sys.path.append("C:/hadoop-3.3.6/bin")

    def initialize_spark_session(self):
        # Initialize spark session
        self.spark = SparkSession.builder\
            .appName("NUS")\
            .config("spark.jars", "mysql-connector-java-8.0.13.jar")\
            .master("local")\
            .getOrCreate()

        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # Read from MySQL Table
    def run_video_upvote_percentage_pipeline(self):
        t0 = time.time()
        print("Running upvote percentage spark pipeline")

        query = f"""
                SELECT SUM(is_upvote) AS upvoted_videos, COUNT(is_upvote) AS number_recommended,
                SUM(is_upvote) / COUNT(is_upvote) AS upvote_percentage, rdv_created_at AS dt
                FROM (
                    SELECT *,
                           IF(t1.v_created_at >= t1.rdv_created_at, 1, 0) AS is_upvote
                    FROM (
                        SELECT rdv.recommendation_id, rdv.user_id, rdv.recommended_video_id, v.video_id,
                        rdv.created_at AS rdv_created_at,
                               v.created_at as v_created_at
                        FROM rs_daily_video_for_user rdv
                            LEFT JOIN
                            (SELECT video_id, voter_id, created_at FROM vote GROUP BY video_id, voter_id) v
                                            ON rdv.recommended_video_id = v.video_id
                                                AND rdv.user_id = v.voter_id) t1
                            WHERE rdv_created_at >= (SELECT MAX(rdv_created_at) - INTERVAL 3 MONTH)
                    ) t2
                    GROUP BY DATE(rdv_created_at)
                """

        df = self.spark.read \
            .format("jdbc") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("url", self.url) \
            .option("query", query) \
            .option("numPartitions", 100) \
            .option("user", config('DB_USER')) \
            .option("password", config('DB_PASSWORD')) \
            .load()

        # write data into new table for faster querying for dashboard
        df.write\
            .format("jdbc") \
            .mode("overwrite") \
            .option("truncate", "true") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("url", self.url) \
            .option("dbtable", "nus_rs_video_upvote") \
            .option("user", config('DB_USER')) \
            .option("password", config('DB_PASSWORD')) \
            .save()

        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} seconds")

    def close_spark_session(self):
        self.spark.stop()
