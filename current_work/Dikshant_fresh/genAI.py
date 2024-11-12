import google.generativeai as genai


LLM_API_KEY = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'

genai.configure(api_key=LLM_API_KEY)

result = genai.embed_content(
    model="models/text-embedding-004",
    content="What is the meaning of life?",
    task_type="retrieval_document",
    title="Embedding of single string"
)

print(len(result['embedding']))