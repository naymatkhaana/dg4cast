

import os
import psycopg2

# Set an environment variable
os.environ["DB_NAME"] = "db_rag"
print("environment variable db_name: ", os.getenv("DB_NAME"))


def create_db():
    db_config = {
        "user": "fs47816", #os.getenv("DB_USER"),
        #"password": "postgres", #os.getenv("DB_PASSWORD"),
        "host":  "localhost", #os.getenv("DB_HOST"),
        "port":  "5432", #os.getenv("DB_PORT"),
        "dbname":  "testdb", #os.getenv("DB_PORT"),
    }

    conn = psycopg2.connect(**db_config)
    conn.autocommit = True  # Enable autocommit for creating the database

    cursor = conn.cursor()
    cursor.execute(
        f"SELECT 1 FROM pg_database WHERE datname = '{os.getenv('DB_NAME')}';"
    )
    database_exists = cursor.fetchone()
    cursor.close()

    if not database_exists:
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE {os.getenv('DB_NAME')};")
        cursor.close()
        print("Database created.")

    conn.close()

    db_config["dbname"] = os.getenv("DB_NAME")
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True

    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.close()

    cursor = conn.cursor()
    #cursor.execute(
    #    "DROP TABLE embeddings;"
    #)
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (id serial PRIMARY KEY, idx_ts integer, col_ts text, parent_ts text, enddate timestamp, embeddings vector(36));"
    )
    cursor.close()

    print("Database setup completed.")

create_db()
