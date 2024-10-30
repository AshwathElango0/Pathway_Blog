import os
from pathway.xpacks.llm.vector_store import VectorStoreClient 
from pathway.xpacks.llm.llms import LiteLLMChat, prompt_chat_single_qa
import litellm

VECTOR_STORE_HOST = "127.0.0.1"
VECTOR_STORE_PORT = 8755
FIXED_QUERY = "Who got nobel prize this time?"
LLM_API_KEY = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"



client = VectorStoreClient(host=VECTOR_STORE_HOST, port=VECTOR_STORE_PORT)


model = LiteLLMChat(
    model_locator="gemini/gemini-1.5-flash",
    api_key=LLM_API_KEY,
    temperature=0.0,
    max_tokens=100,
)


documents = client(FIXED_QUERY, k=1)
print(f"Retrieved {len(documents)} documents.")
# print(documents)

context = "\n\n".join(doc.get('text', '') for doc in documents)


prompt = f"Context:\n{context}\n\nQuestion: {FIXED_QUERY}\n"
res = prompt_chat_single_qa(prompt)
print(f'\n\nResponse: {res}')
response = model(prompt_chat_single_qa(prompt))

print("\n--- LLM Response ---")
print(response)
