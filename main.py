import os
import pathway as pw
from pathway.xpacks.llm.llms import LiteLLMChat, prompt_chat_single_qa

HTTP_HOST = os.environ.get("PATHWAY_REST_CONNECTOR_HOST", "127.0.0.1")
HTTP_PORT = os.environ.get("PATHWAY_REST_CONNECTOR_PORT", "8080")

API_KEY = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"
#  Specific model from OpenAI. You can also use gpt-3.5-turbo for faster responses.
MODEL_LOCATOR = "gemini/gemini-1.5-flash"
# Controls the stochasticity of the openai model output.
TEMPERATURE = 0.0
# Max completion tokens
MAX_TOKENS = 50

class QueryInputSchema(pw.Schema):
    user: str
    query: str

query, response_writer = pw.io.http.rest_connector(
    host=HTTP_HOST,
    port=int(HTTP_PORT),
    schema=QueryInputSchema,
    autocommit_duration_ms=50,
)

pw.set_license_key("demo-license-key-with-telemetry")

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./sample_documents",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

model = LiteLLMChat(
    model="gemini/gemini-1.5-flash", 
    api_key= API_KEY,
)

response = query.select(
    query_id=pw.this.id, result=model(prompt_chat_single_qa(pw.this.query))
)

response_writer(response)
pw.run()