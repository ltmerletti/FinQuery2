import bs4


def is_table_functionally_empty(table_htmll: str) -> bool:
    if not table_htmll:
        return True

    soup = bs4.BeautifulSoup(table_htmll, 'html.parser')

    data_cells = soup.find_all('td')

    for cell in data_cells:
        if cell.get_text(strip=True):
            return False

    return True