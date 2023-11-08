import sys
import os
import time
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.functions import when, sum, count
from decouple import config


class SparkPipeline:
    url = f"jdbc:mysql://{config('DB_HOST')}:{config('DB_PORT')}/{config('DB_NAME')}?useSSL=false"
    spark: SparkSession

    def __init__(self):
        # Initialize HADOOP_HOME environment variable for spark
        os.environ['HADOOP_HOME'] = "C:/hadoop-3.3.6"
        sys.path.append("C:/hadoop-3.3.6/bin")

    def initialize_spark_session(self):
        conf = SparkConf().setAll([("spark.dynamicAllocation.enabled", "true"), ("spark.executor.cores", 5),
                                   ("spark.executor.instances", 3),
                                   # ("spark.dynamicAllocation.minExecutors", "1"),
                                   # ("spark.dynamicAllocation.maxExecutors", "5"),
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

        # print(self.spark.sparkContext.defaultParallelism)

    # Read from MySQL Table
    def run_video_upvote_percentage_pipeline(self):
        """
        Checks if table is updated with the latest date, if it is then
        Queries recommended videos table and vote table to get percentage of videos recommended upvoted
        by the same user, then writes into nus_rs_video_upvote

        Window: 7 days
        """
        # TODO change date to current date in future
        if self.get_max_date("nus_rs_video_upvote") < '2023-10-25':
            t0 = time.time()
            print("Running video upvote percentage spark pipeline")

            query = f"""
                    SELECT SUM(is_upvote) AS upvoted_videos, COUNT(is_upvote) AS number_recommended,
                    SUM(is_upvote) / COUNT(is_upvote) AS upvote_percentage, CAST(DATE(rdv_created_at) AS char) AS dt
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
    --                             WHERE rdv_created_at >= (CURRENT_DATE - INTERVAL 7 DAY)
                                WHERE rdv_created_at >= ('2023-10-25' - INTERVAL 7 DAY)
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

            # .option("numPartitions", 1000) \

            # write data into new table for faster querying for dashboard
            # .option("truncate", "true") \
            df.write \
                .format("jdbc") \
                .mode("overwrite") \
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
        Checks if table is updated with the latest date, if it is then
        Queries recommended conversations table and vote table to get percentage of conversations
        recommended liked by the same user, then writes into nus_rs_conversation_like

        Window: 7 days
        """
        # TODO change date to current date in future
        if self.get_max_date("nus_rs_conversation_like") < '2023-10-25':
            t0 = time.time()
            print("Running conversation like percentage spark pipeline")

            query = f"""
                    SELECT SUM(is_like) AS liked_conversations, COUNT(is_like) AS number_recommended,
                    SUM(is_like) / COUNT(is_like) AS like_percentage, CAST(DATE(rdc_created_at) AS char) AS dt
                    FROM (
                        SELECT *,
                               IF(t1.c_created_at >= t1.rdc_created_at, 1, 0) AS is_like
                        FROM (
                            SELECT rdc.recommended_conversation_id, rdc.user_id,  c.conversation_id,
                            rdc.created_at AS rdc_created_at, c.timestamp AS c_created_at
                            FROM rs_daily_conversation_for_user rdc
                                LEFT JOIN
                                (SELECT conversation_id, like_giver_id, timestamp
                                FROM conversation_like) c
                                                ON rdc.recommended_conversation_id = c.conversation_id
                                                    AND rdc.user_id = c.like_giver_id) t1
    --                             WHERE rdc_created_at >= (CURRENT_DATE - INTERVAL 7 DAY)
                                WHERE rdc_created_at >= ('2023-10-25' - INTERVAL 7 DAY)
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

            # .option("numPartitions", 1000) \

            # write data into new table for faster querying for dashboard
            # .option("truncate", "true") \
            df.write \
                .format("jdbc") \
                .mode("overwrite") \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("url", self.url) \
                .option("dbtable", "nus_rs_conversation_like") \
                .option("user", config('DB_USER')) \
                .option("password", config('DB_PASSWORD')) \
                .save()

            t1 = time.time()
            print(f"Time taken: {t1 - t0:.2f} seconds")

    def get_max_date(self, table_name):
        """
        helper function
        queries table to obtain the latest date
        :param table_name:
        :return: latest date
        """
        max_date = self.spark.read \
            .format("jdbc") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("url", self.url) \
            .option("query", f"SELECT MAX(dt) FROM {table_name}") \
            .option("user", config('DB_USER')) \
            .option("password", config('DB_PASSWORD')) \
            .load() \
            .first()['MAX(dt)']

        return str(max_date)

    def close_spark_session(self):
        self.spark.stop()

    # def test(self):
    #     t0 = time.time()
    #     print("Test spark pipeline")
    #
    #     video_df = self.spark.read \
    #         .format("jdbc") \
    #         .option("driver", "com.mysql.cj.jdbc.Driver") \
    #         .option("url", self.url) \
    #         .option("query", "select user_id, recommended_video_id, date(created_at) as rdv_created_at "
    #                          "from rs_daily_video_for_user "
    #                          "where date(created_at) >= ('2023-10-26' - INTERVAL 7 DAY)") \
    #         .option("user", config('DB_USER')) \
    #         .option("password", config('DB_PASSWORD')) \
    #         .load().repartition(1000, "user_id", "recommended_video_id")
    #
    #     vote_df = self.spark.read \
    #         .format("jdbc") \
    #         .option("driver", "com.mysql.cj.jdbc.Driver") \
    #         .option("url", self.url) \
    #         .option("query", "select video_id, voter_id, date(created_at) as v_created_at from vote "
    #                          "where date(created_at) >= ('2023-10-26' - INTERVAL 7 DAY)") \
    #         .option("user", config('DB_USER')) \
    #         .option("password", config('DB_PASSWORD')) \
    #         .load().dropDuplicates(["video_id", "voter_id"]) \
    #         .repartition(1000, "video_id", "voter_id")
    #
    #     result_df = video_df.join(vote_df, (video_df.user_id == vote_df.voter_id)
    #                               & (video_df.recommended_video_id == vote_df.video_id), 'left') \
    #         .withColumn("is_upvote", when(vote_df.v_created_at >= video_df.rdv_created_at, 1).otherwise(0)) \
    #         .select("is_upvote", "rdv_created_at") \
    #         .groupBy("rdv_created_at").agg(count("is_upvote"), sum("is_upvote"))
    #
    #     video_df.head()
    #     t1 = time.time()
    #     print(f"Time taken: {t1 - t0:.2f} seconds")
