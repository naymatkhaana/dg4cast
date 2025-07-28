import os
import psycopg2


def get_connection():
    conn = psycopg2.connect(
        dbname="db_rag", #os.getenv("DB_NAME"),
        user="fs47816", #os.getenv("DB_USER"),
        #password="postgres", #os.getenv("DB_PASSWORD"),
        host="localhost", #os.getenv("DB_HOST"),
        port="5432", #os.getenv("DB_PORT"),
    )

    return conn