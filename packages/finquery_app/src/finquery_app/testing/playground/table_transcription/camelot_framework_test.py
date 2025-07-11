import camelot as cam

pdf = cam.read_pdf("../../test_docs/aapl-20230930.pdf", flavor='stream')
print(pdf[0].df)

"""
                                               0  1  ...  5              6
0                                                    ...                  
1                                                    ...     September 25,
2                                                    ...              2021
3                                     Net sales:     ...                  
4                                       Products  $  ...  $        297,392
5                                       Services     ...            68,425
6                                Total net sales     ...           365,817
7                                 Cost of sales:     ...                  
8                                       Products     ...           192,266
9                                       Services     ...            20,715
10                           Total cost of sales     ...           212,981
11                                  Gross margin     ...           152,836
12                           Operating expenses:     ...                  
13                      Research and development     ...            21,914
14           Selling, general and administrative     ...            21,973
15                      Total operating expenses     ...            43,887
16                              Operating income     ...           108,949
17                   Other income/(expense), net     ...               258
18      Income before provision for income taxes     ...           109,207
19                    Provision for income taxes     ...            14,527
20                                    Net income  $  ...  $         94,680
21                           Earnings per share:     ...                  
22                                         Basic  $  ...  $           5.67
23                                       Diluted  $  ...  $           5.61
24  Shares used in computing earnings per share:     ...                  
25                                         Basic     ...        16,701,272
26                                       Diluted     ...        16,864,919

[27 rows x 7 columns]
"""