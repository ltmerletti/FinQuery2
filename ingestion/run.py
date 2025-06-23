import time
from add_chunks_to_chroma import add_chunks_to_collection
from chromainit.chromainit import initialize_chroma
from find_file_paths import get_file_paths
from split_into_chunks import load_pdf


# this one may take a while as it parses each pdf. indexing will be added in the future.
def main():
    # set up the collection, or get it if it exists
    print("Connecting to and/or initializing ChromaDB...")
    collection = initialize_chroma()

    # find all the pdfs in the reports folder
    file_paths = get_file_paths("../reports")
    if file_paths:
        print("Files found")
        for file in file_paths:
            print(file)
    else:
        print("No files found")

    # loop through each pdf, chunk it, and add it to the database
    for path in file_paths:
        print(f"\n========================================")
        print(f"STARTING PROCESSING FOR: {path.name}")
        print(f"========================================")

        # load the pdf and split it into chunks
        chunks = load_pdf(path)
        print("Finished loading file")

        # add the chunks to the chroma database
        if chunks:
            add_chunks_to_collection(collection, chunks)
            print(f"Chunks added to collection from {path.name}")
        else:
            print(f"No chunks were generated for {path.name}")

    # all done
    print("\n--- Finished processing all files ---")
    final_count = collection.count()
    print(f"The collection now has {final_count} total chunks")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
