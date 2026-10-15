import re

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents



#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#Drop bibliography chunks: with a small k they crowd real content out of retrieval
_REFERENCE_MARKERS = re.compile(
    r"\bISBN\b|\bdoi\b|PMID|PMC\d|S2CID|OCLC|Archived from the original|Retrieved \d|\bISSN\b",
    re.IGNORECASE,
)


def drop_reference_chunks(chunks, min_markers=2):
    kept = [c for c in chunks if len(_REFERENCE_MARKERS.findall(c.page_content)) < min_markers]
    return kept


#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
