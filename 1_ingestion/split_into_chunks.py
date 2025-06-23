from unstructured.partition.auto import partition


def load_pdf(pdf_file_path):
    print(f"Partitioning document: {pdf_file_path}")

    elements = partition(pdf_file_path, strategy="hi_res")

    print("\n--- Found Elements ---")

    for i, el in enumerate(elements):
        category = el.metadata.category

        text = el.text

        print(f"Element #{i + 1}:")
        print(f"  Category: {category}")
        print(f"  Text: {text[:200]}...")
        print("-" * 20)

    print("\n\n".join([str(el) for el in elements]))
