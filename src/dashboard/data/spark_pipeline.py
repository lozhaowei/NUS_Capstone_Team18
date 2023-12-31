import sys
import os
import time
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from decouple import config


class SparkPipeline:
    url = f"jdbc:mysql://{config('DB_HOST')}:{config('DB_PORT')}/{config('DB_NAME')}?useSSL=false"
    spark: SparkSession

    def __init__(self):
        # Initialize HADOOP_HOME environment variable for spark
        os.environ['HADOOP_HOME'] = "C:/hadoop-3.3.6"
        sys.path.append("C:/hadoop-3.3.6/bin")

    def initialize_spark_session(self):
        print('Starting spark session')

        # Spark Config
        conf = SparkConf().setAll([("spark.dynamicAllocation.enabled", "true"), ("spark.executor.cores", 5),
                                   ("spark.executor.instances", 3),
                                   ("spark.sql.shuffle.partitions", 500),
                                   ("spark.sql.inMemoryColumnarStorage.compressed", "true"),
                                   ("spark.sql.adaptive.skewJoin.enabled", "true")])

        # Initialize spark session
        self.spark = SparkSession.builder \
            .appName("NUS") \
            .config("spark.jars", "mysql-connector-java-8.0.13.jar") \
            .config(conf=conf) \
            .master("local[*]") \
            .getOrCreate()

    # Read from MySQL Table
    def run_video_upvote_percentage_pipeline(self):
        """
        Queries recommended videos table and vote table to get percentage of videos
        recommended upvoted by the same user, then writes into nus_rs_video_upvote
        Only upvotes cast within 2 days of the recommendation are counted

        Window: 2 days
        Start Date: Current date
        """
        t0 = time.time()
        print("Running video upvote percentage spark pipeline")

        query = f"""
                SELECT SUM(is_upvote) AS upvoted_videos, COUNT(is_upvote) AS number_recommended,
                SUM(is_upvote) / COUNT(is_upvote) AS upvote_percentage, 
                CAST(DATE(rdv_created_at) AS char) AS recommendation_date, CURRENT_TIMESTAMP AS dt
                FROM (
                    SELECT *,
                           IF(t1.v_created_at BETWEEN t1.rdv_created_at 
                           AND DATE_ADD(t1.rdv_created_at, INTERVAL 2 DAY), 1, 0) AS is_upvote
                    FROM (
                        SELECT rdv.recommendation_id, rdv.user_id, rdv.recommended_video_id, v.video_id,
                        rdv.created_at AS rdv_created_at,
                               v.created_at as v_created_at
                        FROM rs_daily_video_for_user rdv
                            LEFT JOIN
                            (SELECT video_id, voter_id, created_at FROM vote GROUP BY video_id, voter_id) v
                                            ON rdv.recommended_video_id = v.video_id
                                                AND rdv.user_id = v.voter_id) t1
                            WHERE rdv_created_at >= (CURRENT_DATE - INTERVAL 2 DAY)
                    ) t2
                    GROUP BY DATE(rdv_created_at)
                """

        df = self.spark.read \
            .format("jdbc") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("url", self.url) \
            .option("query", query) \
            .option("user", config('DB_USER')) \
            .option("password", config('DB_PASSWORD')) \
            .load()

        # write data into new table for faster querying for dashboard
        df.write \
            .format("jdbc") \
            .mode("append") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("url", self.url) \
            .option("dbtable", "nus_rs_video_upvote") \
            .option("user", config('DB_USER')) \
            .option("password", config('DB_PASSWORD')) \
            .save()

        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} seconds")

    def run_conversation_like_percentage_pipeline(self):
        """
        Queries recommended conversations table and vote table to get percentage of conversations
        recommended liked by the same user, then writes into nus_rs_conversation_like
        Only likes cast within 2 days of the recommendation are counted

        Window: 2 days
        Start Date: Current date
        """
        t0 = time.time()
        print("Running conversation like percentage spark pipeline")

        query = f"""
                SELECT SUM(is_like) AS liked_conversations, COUNT(is_like) AS number_recommended,
                SUM(is_like) / COUNT(is_like) AS like_percentage, 
                CAST(DATE(rdc_created_at) AS char) AS recommendation_date, CURRENT_TIMESTAMP AS dt
                FROM (
                    SELECT *,
                           IF(t1.c_created_at BETWEEN t1.rdc_created_at 
                           AND DATE_ADD(t1.rdc_created_at, INTERVAL 2 DAY), 1, 0) AS is_like
                    FROM (
                        SELECT rdc.recommended_conversation_id, rdc.user_id,  c.conversation_id,
                        rdc.created_at AS rdc_created_at, c.timestamp AS c_created_at
                        FROM rs_daily_conversation_for_user rdc
                            LEFT JOIN
                            (SELECT conversation_id, like_giver_id, timestamp
                            FROM conversation_like) c
                                            ON rdc.recommended_conversation_id = c.conversation_id
                                                AND rdc.user_id = c.like_giver_id) t1
                            WHERE rdc_created_at >= (CURRENT_DATE - INTERVAL 2 DAY)
                    ) t2
                    GROUP BY DATE(rdc_created_at)
                """

        df = self.spark.read \
            .format("jdbc") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("url", self.url) \
            .option("query", query) \
            .option("user", config('DB_USER')) \
            .option("password", config('DB_PASSWORD')) \
            .load()

        # write data into new table for faster querying for dashboard
        df.write \
            .format("jdbc") \
            .mode("append") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("url", self.url) \
            .option("dbtable", "nus_rs_conversation_like") \
            .option("user", config('DB_USER')) \
            .option("password", config('DB_PASSWORD')) \
            .save()

        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} seconds")

    def close_spark_session(self):
        print('Closing spark session')
        self.spark.stop()
