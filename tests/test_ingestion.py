from io import BytesIO

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from studybuddy.ingestion import load_uploaded_file, split_documents, stable_document_id


def test_txt_loads_utf8_and_source_metadata():
    docs = load_uploaded_file("aula.txt", "Olá mundo".encode())
    assert docs[0].page_content == "Olá mundo"
    assert docs[0].metadata["source"] == "aula.txt"


def _pdf_with_pages(*texts: str) -> bytes:
    """Monta um PDF mínimo com uma linha de texto selecionável por página."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in texts:
        stream = f"BT /F1 12 Tf 20 50 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_ref = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents {content_ref} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    output = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    output += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return output


def test_pdf_keeps_page_numbers():
    docs = load_uploaded_file("aula.pdf", _pdf_with_pages("Primeira pagina", "Segunda pagina"))
    assert [doc.metadata["page"] for doc in docs] == [1, 2]
    assert "Segunda" in docs[1].page_content
    assert all(doc.metadata["source"] == "aula.pdf" for doc in docs)


def test_blank_pdf_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="Nenhum texto"):
        load_uploaded_file("vazio.pdf", output.getvalue())


def test_rejects_unsupported_files():
    with pytest.raises(ValueError, match="Formato não suportado"):
        load_uploaded_file("arquivo.docx", b"x")


def test_chunks_preserve_metadata_and_add_index():
    docs = split_documents([Document(page_content="palavra " * 100, metadata={"source": "x.txt"})], 100, 10)
    assert len(docs) > 1
    assert all(doc.metadata["source"] == "x.txt" for doc in docs)
    assert [doc.metadata["chunk"] for doc in docs] == list(range(len(docs)))


def test_stable_id_changes_by_user_and_content():
    assert stable_document_id("u1", "a.txt", b"x") != stable_document_id("u2", "a.txt", b"x")
    assert stable_document_id("u1", "a.txt", b"x") != stable_document_id("u1", "a.txt", b"y")
