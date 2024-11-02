# retriever.py
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
SERVER_HOST = os.getenv("VECTOR_STORE_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("VECTOR_STORE_PORT", 8000))
QUERY_ENDPOINT = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/pw_ai_answer"



def send_query(query):
    """
    Sends a query to the RAG server and returns the response.
    
    Args:
        query (str): The user's query string.
    
    Returns:
        dict: The server's JSON response.
    """
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": query
    }
    
    try:
        response = requests.post(QUERY_ENDPOINT, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the server. Please ensure the server is running.")
    except requests.exceptions.Timeout:
        print("Error: The request timed out.")
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")
    return None

def main():
    print("=== Pathway RAG Retriever ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Enter your query: ").strip()
        if query.lower() == 'exit':
            print("Exiting the retriever. Goodbye!")
            break
        if not query:
            print("Please enter a valid query.\n")
            continue
        
        print("\nSending query to the server...\n")
        response = send_query(query)
        
        if response:
            print("=== Retrieved Documents ===")
            for idx, doc in enumerate(response.get("retrieved_documents", []), start=1):
                print(f"\nDocument {idx}:")
                print(doc.get("text", "No text available."))
                # If there is metadata, you can display it as well
                if doc.get("metadata"):
                    print(f"Metadata: {doc.get('metadata')}")
            
            print("\n=== LLM Response ===")
            print(response.get("response", "No response generated."))
            print("\n" + "-"*50 + "\n")
        else:
            print("Failed to retrieve a valid response from the server.\n")

if __name__ == "__main__":
    main()
