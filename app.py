import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
import pinecone

from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-bot")
MODEL_PATH = os.environ.get("MODEL_PATH", "model/llama-2-7b-chat.ggmlv3.q4_0.bin")

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"Model file not found: {MODEL_PATH}\n"
        "Download it first; see model/instruction.txt."
    )

embeddings = download_hugging_face_embeddings()

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Low temperature: grounded extraction does not want creativity.
llm = CTransformers(
    model=MODEL_PATH,
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.2},
)

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
    result = qa({"query": msg})
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
