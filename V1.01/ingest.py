import os
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import WeaviateDocumentStore
from haystack.preview.components.file_converters.pypdf import PyPDFToDocument
from haystack import Pipeline

print("Import Successfully")

path_doc = ["PdfOutput/" + f for f in os.listdir("PdfOutput") if f.endswith('.pdf')]

document_store = WeaviateDocumentStore(host='http://localhost',
                                       port=8080,
                                       embedding_dim=768)

print("Document Store: ", document_store)
print("#####################")

converter = PyPDFToDocument()
print("Converter: ", converter)
print("#####################")
output = converter.run(paths=path_doc)
docs = output["documents"]
print("Docs: ", docs)
print("#####################")

final_docs = []
for doc in docs:
    print(doc.text)
    new_doc = {
        'content': doc.text,
        'meta': doc.metadata
    }
    final_docs.append(new_doc)
    print("#####################")

# No need for preprocessor since we are not splitting documents
print("Final Docs: ", final_docs)
print("#####################")

# Write the full documents to the document store
document_store.write_documents(final_docs)

# Initialize the retriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

print("Retriever: ", retriever)

# Update embeddings for the full documents
document_store.update_embeddings(retriever)

print("Embeddings Done.")
