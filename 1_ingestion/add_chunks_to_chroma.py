# We will always use ChromaDB upsert rather than insert to avoid duplication
# I will still hold a mysql database with the document contents to double-check
# https://python.langchain.com/docs/how_to/indexing/ along with postgres
