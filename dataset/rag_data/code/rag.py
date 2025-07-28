from itertools import chain
import torch
from pgvector.psycopg2 import register_vector
from db import get_connection

from pgvector.psycopg2 import register_vector
import pandas as pd
from sklearn.preprocessing import StandardScaler

template = """[INST]
You are a friendly documentation search bot.
Use following piece of context to answer the question.
If the context is empty, try your best to answer without it.
Never mention the context.
Try to keep your answers concise unless asked to provide details.

Context: {context}
Question: {question}
[/INST]
Answer:
"""

def get_retrieval_condition(query_embedding, enddate, threshold=0.7):
    # Convert query embedding to a string format for SQL query
    query_embedding_str = ",".join(map(str, query_embedding))

    # SQL condition for cosine similarity
    condition = f"(embeddings <=> '[{query_embedding_str}]') < {threshold} and enddate < '{enddate}' ORDER BY embeddings <=> '[{query_embedding_str}]'"
    return condition


def rag_query( query):
    # Generate query embedding
    #query_embedding = generate_embeddings(
    #   tokenizer=tokenizer, model=model, device=device, text=query
    #)[1]

    scaler = StandardScaler()       
    df = pd.read_csv(query)
    cols = list(df.columns)
    cols.remove("date")
    
    query_embedding = df[cols[-3]].to_numpy()[985:985+36]
    enddate = str(df['date'][985+36 -1])
    query_embedding = scaler.fit_transform(query_embedding.reshape(-1,1)).reshape(-1,)

    # Retrieve relevant embeddings from the database
    retrieval_condition = get_retrieval_condition(query_embedding,enddate)

    conn = get_connection()
    register_vector(conn)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT idx_ts, col_ts, parent_ts, enddate FROM embeddings WHERE {retrieval_condition} LIMIT 20"
    )
    
    print(f"SELECT idx_ts, col_ts, parent_ts, enddate FROM embeddings WHERE {retrieval_condition} LIMIT 20")
    retrieved = cursor.fetchall()

    rag_query = ' '.join([str(row[0])+" "+str(row[1])+" "+str(row[2])+" "+str(row[3])+"\n" for row in retrieved])
    print(rag_query)

    query_template = template.format(context=rag_query, question=query)

    #input_ids = tokenizer.encode(query_template, return_tensors="pt")

    # Generate the response
    #generated_response = model.generate(input_ids.to(device), max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    return retrieved[0][0],retrieved[0][1],retrieved[0][2] #tokenizer.decode(generated_response[0][input_ids.shape[-1]:], skip_special_tokens=True)

