from unstructured.partition.pdf import partition_pdf

pdf_path = "../../test_docs/aapl-20230930-short.pdf"

elements = list(partition_pdf(
    pdf_path, infer_table_structure=True, strategy="hi_res"
))

for i, el in enumerate(elements):
    print(i, ": ", el.text)

"""
0 :  6/11/25, 11:56 AM
1 :  aapl-20230930
2 :  Apple Inc.
3 :  CONSOLIDATED STATEMENTS OF OPERATIONS
4 :  (In millions, except number of shares, which are reﬂected in thousands, and per-share amounts)
5 :  Years ended September 30, September 24, 2023 2022 $ 298,085 $ 316,199 $ 85,200 78,129 Total net sales 383,285 394,328 189,282 201,471 24,855 22,075 Total cost of sales 214,137 223,546 Gross margin 169,148 170,782 Research and development 29,915 26,251 Selling, general and administrative 24,932 25,094 Total operating expenses 54,847 51,345 114,301 119,437 (565) (334) 113,736 119,103 16,741 19,300 $ 96,995 $ 99,803 $ Basic $ 6.16 $ 6.15 $ Diluted $ 6.13 $ 6.11 $ Basic 15,744,231 16,215,963 Diluted 15,812,547 16,325,819 September 25, 2021
6 :  Net sales:
7 :  Products
8 :  Services
9 :  Cost of sales:
10 :  Products
11 :  Services
12 :  Operating expenses:
13 :  Operating income
14 :  Other income/(expense), net
15 :  Income before provision for income taxes
16 :  Provision for income taxes
17 :  Net income
18 :  Earnings per share:
19 :  Shares used in computing earnings per share:
20 :  See accompanying Notes to Consolidated Financial Statements.
21 :  Apple Inc. | 2023 Form 10-K | 28
22 :  https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm
23 :  297,392
24 :  68,425
25 :  365,817
26 :  192,266
27 :  20,715
28 :  212,981
29 :  152,836
30 :  21,914
31 :  21,973
32 :  43,887
33 :  108,949
34 :  258
35 :  109,207
36 :  14,527
37 :  94,680
38 :  5.67
39 :  5.61
40 :  16,701,272
41 :  16,864,919
42 :  40/74
"""
