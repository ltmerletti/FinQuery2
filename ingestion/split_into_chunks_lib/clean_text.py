import re


def clean_element_text(textt: str) -> str:
    # removes urls (we don't need them for the type of query)
    textt = re.sub(r'https?://\S+', '', textt, flags=re.MULTILINE)
    # removes sec urls
    textt = re.sub(r'\S*www\.sec\.gov\S*', '', textt, flags=re.MULTILINE)
    # remove page number (which is in format (a number)/(a number)
    textt = re.sub(r'\s*\d+/\d+\s*', '', textt)
    # removes dates and times
    textt = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}(,\s*\d{1,2}:\d{2}\s*(AM|PM)?)?', '', textt)
    # removes stray page numbers
    textt = re.sub(r'^\s*\d+\s*$', '', textt, flags=re.MULTILINE)
    # removes empty lines
    textt = re.sub(r'\n\s*\n', '\n', textt)

    return textt.strip()


def clean_table_text(textt: str) -> str:
    # removes empty lines
    textt = re.sub(r'\n\s*\n', '\n', textt)

    return textt.strip()
