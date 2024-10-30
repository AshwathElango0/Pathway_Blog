import os
from pathway.xpacks.llm.vector_store import VectorStoreClient 
from pathway.xpacks.llm.llms import LiteLLMChat, prompt_chat_single_qa
import litellm
# Configuration
VECTOR_STORE_HOST = "127.0.0.1"
VECTOR_STORE_PORT = 8755
FIXED_QUERY = "Who got nobel prize this time?"
LLM_API_KEY = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"


# Initialize VectorStoreClient
client = VectorStoreClient(host=VECTOR_STORE_HOST, port=VECTOR_STORE_PORT)

# Initialize LLM
model = LiteLLMChat(
    model_locator="gemini/gemini-1.5-flash",
    api_key=LLM_API_KEY,
    temperature=0.0,
    max_tokens=100,
)

# Fetch documents
documents = client(FIXED_QUERY, k=1)
print(f"Retrieved {len(documents)} documents.")
# print(documents)
# Prepare context
context = "\n\n".join(doc.get('text', '') for doc in documents)

# Create prompt and get response
prompt = f"Context:\n{context}\n\nQuestion: {FIXED_QUERY}\n"
res = prompt_chat_single_qa(prompt)
print(f'\n\nResponse: {res}')
response = model(prompt_chat_single_qa(prompt))

print("\n--- LLM Response ---")
print(response)
