import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler

from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

app = Flask(__name__)

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "medical-bot")

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY is not set. See .env.example.")

# Langfuse traces every chain/LLM call (retrieval + generation) when configured;
# the app still runs with tracing off if the keys are absent.
langfuse_handler = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    langfuse_handler = CallbackHandler()

if not os.path.isdir(CHROMA_DIR):
    raise SystemExit(
        f"Chroma index not found at {CHROMA_DIR}.\n"
        "Build it first: python store_index.py"
    )

embeddings = download_hugging_face_embeddings()
docsearch = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Low temperature: grounded extraction does not want creativity.
llm = ChatGroq(model=GROQ_MODEL, temperature=0.2, api_key=GROQ_API_KEY)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = (request.form.get("msg") or "").strip()
    if not msg:
        return jsonify({"error": "empty message"}), 400
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    try:
        result = qa.invoke({"query": msg}, config=config)
    except Exception:
        # Groq timeouts, rate limits and network errors land here. Log the cause, answer in JSON.
        app.logger.exception("answering failed")
        return jsonify({"error": "The model could not answer right now. Please try again."}), 502
    sources = [
        {
            "source": os.path.basename(str(d.metadata.get("source", ""))),
            "page": int(d.metadata["page"]) + 1 if "page" in d.metadata else None,
        }
        for d in result["source_documents"]
    ]
    return jsonify({"answer": result["result"].strip(), "sources": sources})


if __name__ == "__main__":
    # The Werkzeug debugger executes arbitrary code; never expose it on 0.0.0.0 by default.
    app.run(host="0.0.0.0", port=8080, debug=os.environ.get("FLASK_DEBUG") == "1")
