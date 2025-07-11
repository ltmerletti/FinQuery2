import pdfplumber
import pandas as pd

# Define the path to your PDF file.
pdf_path = "../../test_docs/aapl-20230930.pdf"

try:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]

        table_data = page.extract_table()

        if table_data:
            df = pd.DataFrame(table_data[1:], columns=table_data[0])

            pd.set_option('display.max_columns', None)
            print("--- Table extracted with pdfplumber ---")
            print(df)
        else:
            print("No table found on the specified page with pdfplumber.")

except Exception as e:
    print(f"An error occurred with pdfplumber: {e}")

"""
--- Table extracted with pdfplumber ---
                                        Products           $ 298,085    \
0                                       Services      85,200    None     
1                                Total net sales     383,285    None     
2                                                               None     
3                                 Cost of sales:                None     
4                                       Products     189,282    None     
5                                       Services      24,855    None     
6                            Total cost of sales     214,137    None     
7                                   Gross margin     169,148    None     
8                                                               None     
9                            Operating expenses:                None     
10                      Research and development      29,915    None     
11           Selling, general and administrative      24,932    None     
12                      Total operating expenses      54,847    None     
13                                                              None     
14                              Operating income     114,301    None     
15                   Other income/(expense), net       (565)    None     
16      Income before provision for income taxes     113,736    None     
17                    Provision for income taxes      16,741    None     
18                                    Net income           $  96,995     
19                                                              None     
20                           Earnings per share:                None     
21                                         Basic           $    6.16     
22                                       Diluted           $    6.13     
23                                                              None     
24  Shares used in computing earnings per share:                None     
25                                         Basic  15,744,231    None     
26                                       Diluted  15,812,547    None     

             $ 316,199             $ 297,392  
0       78,129    None        68,425    None  
1      394,328    None       365,817    None  
2                 None                  None  
3                 None                  None  
4      201,471    None       192,266    None  
5       22,075    None        20,715    None  
6      223,546    None       212,981    None  
7      170,782    None       152,836    None  
8                 None                  None  
9                 None                  None  
10      26,251    None        21,914    None  
11      25,094    None        21,973    None  
12      51,345    None        43,887    None  
13                None                  None  
14     119,437    None       108,949    None  
15       (334)    None           258    None  
16     119,103    None       109,207    None  
17      19,300    None        14,527    None  
18           $  99,803             $  94,680  
19                None                  None  
20                None                  None  
21           $    6.15             $    5.67  
22           $    6.11             $    5.61  
23                None                  None  
24                None                  None  
25  16,215,963    None    16,701,272    None  
26  16,325,819    None    16,864,919    None  
"""