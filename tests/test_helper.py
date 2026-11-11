from langchain_core.documents import Document

from src.helper import drop_reference_chunks


def test_drops_reference_lists_and_keeps_content():
    content = Document(page_content="The common cold is a viral infection of the nose and throat.")
    references = Document(page_content="Smith J. Colds. ISBN 978-0-00-000000-0. doi:10.1000/xyz. PMID 12345.")
    one_citation = Document(page_content="Rhinoviruses cause most colds (atlas: ISBN 978-0-00-000000-0).")

    assert drop_reference_chunks([content, references, one_citation]) == [content, one_citation]
