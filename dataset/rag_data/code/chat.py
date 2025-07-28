from rag import rag_query


def chat():
    print("Chat started") #"Chat started. Type 'exit' to end the chat.")

    while True:
        question = input("Ask a question: (Type filename of time series to compare or type 'exit' to end the chat.) ")

        if question.lower() == "exit":
            break

        idx_ts, col_ts, parent_ts = rag_query(query=question)

        print(f"You Asked: {question} (filename of time series) taking last patch/chunk/window to compare")
        print("Answer: idx_ts, col_ts, parent_ts, embedding_ts", idx_ts, col_ts, parent_ts)

    print("Chat ended.")
    
chat()