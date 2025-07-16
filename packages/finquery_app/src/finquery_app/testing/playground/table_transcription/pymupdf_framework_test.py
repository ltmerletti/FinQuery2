import fitz
import pandas as pd
from collections import defaultdict

pdf_path = "../../test_docs/aapl-20230930.pdf"

table_bbox = (40, 150, 580, 700)

try:
    doc = fitz.open(pdf_path)

    page = doc.load_page(0)

    words = page.get_text("words")

    table_words = [w for w in words if fitz.Rect(w[:4]).intersects(table_bbox)]

    if not table_words:
        print("No words found within the specified bounding box.")
    else:
        y_tolerance = 4
        lines = defaultdict(list)
        for w in table_words:
            found_line = False
            for y_val in lines.keys():
                if abs(w[1] - y_val) < y_tolerance:
                    lines[y_val].append(w)
                    found_line = True
                    break
            if not found_line:
                lines[w[1]].append(w)

        sorted_lines = sorted(lines.items())

        reconstructed_rows = []
        for _, line_words in sorted_lines:
            line_words.sort(key=lambda w: w[0])
            reconstructed_rows.append([word[4] for word in line_words])

        df = pd.DataFrame(reconstructed_rows)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("--- Table reconstructed with PyMuPDF ---")
        print(df)

except Exception as e:
    print(f"An error occurred with PyMuPDF: {e}")

"""
--- Table reconstructed with PyMuPDF ---
            0                  1            2               3             4          5            6        7        8
0        2023               2022         2021            None          None       None         None     None     None
1         Net             sales:         None            None          None       None         None     None     None
2    Products                  $      298,085               $       316,199          $      297,392     None     None
3    Services             85,200       78,129          68,425          None       None         None     None     None
4       Total                net        sales         383,285       394,328    365,817         None     None     None
5        Cost                 of       sales:            None          None       None         None     None     None
6    Products            189,282      201,471         192,266          None       None         None     None     None
7    Services             24,855       22,075          20,715          None       None         None     None     None
8       Total               cost           of           sales       214,137    223,546      212,981     None     None
9       Gross             margin      169,148         170,782       152,836       None         None     None     None
10  Operating          expenses:         None            None          None       None         None     None     None
11   Research                and  development          29,915        26,251     21,914         None     None     None
12   Selling,            general          and  administrative        24,932     25,094       21,973     None     None
13      Total          operating     expenses          54,847        51,345     43,887         None     None     None
14  Operating             income      114,301         119,437       108,949       None         None     None     None
15      Other  income/(expense),          net           (565)         (334)        258         None     None     None
16     Income             before    provision             for        income      taxes      113,736  119,103  109,207
17  Provision                for       income           taxes        16,741     19,300       14,527     None     None
18        Net             income            $          96,995             $     99,803            $   94,680     None
19   Earnings                per       share:            None          None       None         None     None     None
20      Basic                  $         6.16               $          6.15          $         5.67     None     None
21    Diluted                  $         6.13               $          6.11          $         5.61     None     None
22     Shares               used           in       computing      earnings        per       share:     None     None
23      Basic         15,744,231   16,215,963      16,701,272          None       None         None     None     None
24    Diluted         15,812,547   16,325,819      16,864,919          None       None         None     None     None
25        See       accompanying        Notes              to  Consolidated  Financial  Statements.     None     None
26      Apple               Inc.            |            2023          Form       10-K            |       28     None
"""