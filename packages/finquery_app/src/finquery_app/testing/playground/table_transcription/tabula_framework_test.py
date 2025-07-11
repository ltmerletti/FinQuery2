import tabula as tb
import pandas as pd

pdf_path = "../../test_docs/aapl-20230930.pdf"

try:
    tables = tb.read_pdf(
        pdf_path,
        area=(150, 40, 700, 580),
        stream=True,
        pandas_options={'header': None}
    )

    if tables:
        df = tables[0]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("--- Table extracted with tabula-py (using specific area) ---")
        print(df)
    else:
        print("No tables found on the specified page with tabula-py.")

except Exception as e:
    print(f"An error occurred with tabula-py: {e}")
    print("Please ensure you have Java installed and that tabula-py is configured correctly.")

"""
--- Table extracted with tabula-py (using specific area) ---
                                                    0           1    2   3           4    5           6
0                                           et sales:         NaN  NaN NaN         NaN  NaN         NaN
1                                          Products $     298,085    $ NaN     316,199    $     297,392
2                                            Services      85,200  NaN NaN      78,129  NaN      68,425
3                                     Total net sales     383,285  NaN NaN     394,328  NaN     365,817
4                                       ost of sales:         NaN  NaN NaN         NaN  NaN         NaN
5                                            Products     189,282  NaN NaN     201,471  NaN     192,266
6                                            Services      24,855  NaN NaN      22,075  NaN      20,715
7                                 Total cost of sales     214,137  NaN NaN     223,546  NaN     212,981
8                                        Gross margin     169,148  NaN NaN     170,782  NaN     152,836
9                                  perating expenses:         NaN  NaN NaN         NaN  NaN         NaN
10                           Research and development      29,915  NaN NaN      26,251  NaN      21,914
11                Selling, general and administrative      24,932  NaN NaN      25,094  NaN      21,973
12                           Total operating expenses      54,847  NaN NaN      51,345  NaN      43,887
13                                    perating income     114,301  NaN NaN     119,437  NaN     108,949
14                         ther income/(expense), net       (565)  NaN NaN       (334)  NaN         258
15             come before provision for income taxes     113,736  NaN NaN     119,103  NaN     109,207
16                          rovision for income taxes      16,741  NaN NaN      19,300  NaN      14,527
17                                        et income $      96,995    $ NaN      99,803    $      94,680
18                                 arnings per share:         NaN  NaN NaN         NaN  NaN         NaN
19                                            Basic $        6.16    $ NaN        6.15    $        5.67
20                                          Diluted $        6.13    $ NaN        6.11    $        5.61
21        hares used in computing earnings per share:         NaN  NaN NaN         NaN  NaN         NaN
22                                              Basic  15,744,231  NaN NaN  16,215,963  NaN  16,701,272
23                                            Diluted  15,812,547  NaN NaN  16,325,819  NaN  16,864,919
24  See accompanying Notes to Consolidated Financi...         NaN  NaN NaN         NaN  NaN         NaN
25                   Apple Inc. | 2023 Form 10-K | 28         NaN  NaN NaN         NaN  NaN         NaN
"""