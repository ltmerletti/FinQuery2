import pathlib
from docling.document_converter import DocumentConverter

sources = [
    pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/reports/added/aapl-20230930.pdf"),
    pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/reports/added/lly-20221231.pdf"),
    pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/reports/added/nflx-20231231.pdf"),
    pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/reports/added/nvda-20230129.pdf"),
    pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/reports/added/pltr-20231231.pdf"),
    pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/reports/added/tsla-20230930.pdf"),
    pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/reports/added/tsla-20240930.pdf")
]

output_dir = pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/packages/finquery_app/src/finquery_app/testing/playground/table_transcription/docling_md_files")

output_dir.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()

for source in sources:
    print(f"Processing: {source.name}")

    result = converter.convert(source)

    markdown_content = result.document.export_to_markdown()

    ticker = source.name[:4]
    output_filename = f"docling_{ticker}_pdf_as_markdown.md"
    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Successfully saved Markdown to: {output_path}")

