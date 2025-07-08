import os
from pprint import pprint

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

from ingestion.chromainit.database_setup import get_embeddings


def setup_langfuse():
    """Initializes and returns the Langfuse callback handler."""
    load_dotenv()
    return CallbackHandler()


def initialize_vector_store(persist_directory="./chromadb", collection_name="financial_documents"):
    return Chroma(collection_name=collection_name, embedding_function=get_embeddings(),
                  persist_directory=persist_directory)


def get_db():
    vector_store = Chroma(collection_name="financial_documents", embedding_function=get_embeddings(),
                          persist_directory="../chromadb"  # Note the different path
                          )
    return vector_store.get()


def execute_query(query_text: str, vectorstore: Chroma, num_to_fetch: int, config: RunnableConfig):
    if not query_text:
        print("Query text cannot be empty.")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={'k': num_to_fetch})

    results = retriever.invoke(query_text, config=config)
    return results


def get_rag_test_questions():
    return [
        # Direct Data Retrieval
        "What were Apple's total net sales in 2023?", "How much did Apple spend on Research and Development in 2023?",
        "What was the net income for fiscal year 2022?",
        "Find the total assets listed on the Consolidated Balance Sheets for 2023.",
        "What were the net sales for the Mac product line in 2023?",  # Comparative & Trend Analysis
        "How did iPhone net sales in 2023 compare to 2022?",
        "Did Services net sales increase or decrease from 2022 to 2023? By how much?",
        "What was the percentage change in Mac net sales between 2023 and 2022?",
        "Compare the net sales of the Americas region versus the Europe region in 2023.",
        "What was the trend for Wearables, Home and Accessories sales over the last three fiscal years shown?",
        # Narrative & Explanatory Retrieval
        "According to the document, why did Mac net sales decrease in 2023?",
        "What factors contributed to the change in total net sales in 2023 compared to the previous year?",
        "What were some of the significant product announcements in the first quarter of fiscal year 2023?",
        "What does the document say about the impact of foreign currency weakness on net sales?",
        "Describe the company's historical seasonality in sales.",  # Deeper & Component-Based Questions
        "What were the net sales for the 'Rest of Asia Pacific' region in 2023?",
        "Break down the Services net sales for 2023 if possible.", "What are the main components of 'Cost of Sales'?",
        "What were the total liabilities and shareholders' equity for 2023?",
        "How much cash and cash equivalents did Apple have at the end of the 2023 fiscal year?",

        # Direct Data Retrieval (Specific Facts & Figures)
        "What is Apple's I.R.S. Employer Identification No.?",
        "On which stock market is Apple's common stock traded, and what is its ticker symbol?",
        "How many full-time equivalent employees did the company have as of September 30, 2023?",
        "What was the total operating income in 2023?", "Find the total 'Other income/(expense), net' for 2022.",
        "How many shares of common stock were issued and outstanding as of October 20, 2023?",
        "What was the effective tax rate for 2023?", "What were the total liabilities as of September 24, 2022?",
        "Identify the net income for the fiscal year 2021.",
        "How much was spent on repurchases of common stock in the fiscal year 2023, according to the Consolidated Statements of Cash Flows?",
        "What were the net sales in the Greater China region for 2022?",
        "Find the gross margin percentage for Products in 2023.",
        "Who is Apple's independent registered public accounting firm?",
        "What was the aggregate market value of the voting and non-voting stock held by non-affiliates as of March 31, 2023?",
        "How many weeks were in fiscal year 2023, and how many were in 2022?",
        "On what date was this Form 10-K report signed by the Chief Financial Officer?",
        "Find the total amount of dividends and dividend equivalents declared in 2023.",
        "What was the total amount of commercial paper outstanding as of September 30, 2023?",
        "What was the par value of Apple's common stock?",
        "What was the total value of inventories on the balance sheet for September 30, 2023?",
        # Comparative & Trend Analysis
        "Compare R&D spending as a percentage of total net sales for 2023 and 2022.",
        "How did diluted earnings per share change between 2021 and 2023?",
        "Compare the operating income of the Americas segment in 2023 versus 2022.",
        "Did total shareholders' equity increase or decrease from the 2022 fiscal year-end to the 2023 fiscal year-end?",
        "How has the Services gross margin percentage evolved from 2022 to 2023?",
        "What is the year-over-year percentage change in net sales for the Japan segment from 2022 to 2023?",
        "Compare the cash generated by financing activities in 2023 with 2022.",
        "Track the change in 'Total shareholders' equity' from the end of fiscal year 2021 to the end of fiscal year 2023.",
        "Based on the stock performance graph, which grew more between September 2020 and September 2023: Apple Inc. stock or the S&P 500 Index?",
        "What was the percentage change in Selling, General and Administrative expenses between 2023 and 2022?",
        "Compare the total cost of sales for Products in 2023 versus 2022.",
        "How did total current assets change from September 24, 2022, to September 30, 2023?",
        "Analyze the trend of cash paid for income taxes from 2021 to 2023.",
        "Compare the number of common shares repurchased in 2023 to the number repurchased in 2021.",
        "How did the weighted-average interest rate on commercial paper as of the end of the 2023 fiscal year compare to the rate at the end of 2022?",
        # Narrative & Explanatory Retrieval
        "What are the primary risks associated with the company's reliance on outsourcing partners for manufacturing?",
        "Describe the key factors that could cause the company's quarterly results to fluctuate.",
        "What does the company state about its competition in the services market, especially regarding free business models?",
        "Summarize the company's approach to human capital management, specifically regarding compensation and benefits.",
        "Explain the 'Masimo' legal proceeding and its status as of the report date.",
        "What are the main distribution channels Apple uses, and what was the sales split between them in 2023?",
        "What risks does the company identify related to its supply of components, especially for new products?",
        "According to the 'Risk Factors' section, how could international trade disputes, such as those between the U.S. and China, impact Apple?",
        "How does the company define its 'Greater China' segment?",
        "What does the report say about the potential impact of macroeconomic conditions like inflation on the business?",
        "Describe the company's strategy regarding obtaining components from single or limited sources.",
        "What are the principal competitive factors that are important to the company?",
        "What is the company's policy on intellectual property protection, and how does it rely on it?",
        "What were some of the significant product announcements in the third quarter of fiscal year 2023?",
        "Explain the risks associated with information technology system failures or network disruptions.",
        # Deeper, Component-Based & Inferential Questions
        "Calculate the gross margin percentage for the Services segment in 2022.",
        "What was the total amount of long-term debt (non-current portion) as of September 30, 2023?",
        "Identify the total unrecognized compensation cost related to RSUs as of September 30, 2023, and over what period is it expected to be recognized?",
        "What were the main drivers for the increase in R&D expenses in 2023?",
        "From the Consolidated Statements of Cash Flows, what was the net change in cash from investing activities in 2023?",
        "In the 'Risk Factors,' what does the company state about its market share in the global smartphone, personal computer, and tablet markets?",
        "How does Apple account for revenue from third-party applications sold through the App Store?",
        "What was the total value of 'Land and buildings' (gross) on the balance sheet at the end of fiscal 2023?",
        "What was the total up-front payment for the Accelerated Share Repurchase agreements (ASRs) entered into in August 2023?",
        "According to the report, what is the potential decline in fair value of the investment portfolio from a hypothetical 100 basis point increase in interest rates in 2023?",
        "What legal risks does Apple associate with the EU Digital Markets Act?",
        "What were the foreign pretax earnings for 2023?",
        "How much of the total deferred revenue as of September 30, 2023, is expected to be realized in less than one year?",
        "Identify the total future principal payments for Notes due in the year 2025.",
        "What were the main reasons for the decrease in the effective tax rate in 2023 compared to 2022?",
        "What was the total amount of future payments for noncancelable unconditional purchase obligations with a remaining term longer than one year as of September 30, 2023?",
        "What were the components of 'Other income/(expense), net' for 2023?",
        "According to the legal proceedings section, what was the primary ruling and injunction issued by the District Court in the Epic Games case?",
        "How much did the company accrue for manufacturing purchase obligations payable within 12 months as of September 30, 2023?",
        "What was the weighted-average remaining lease term for the company's lease liabilities as of September 30, 2023?",
        "What was the total amount of gross unrecognized tax benefits as of September 30, 2023?",
        "Calculate the company's total liabilities to total assets ratio for the fiscal year ended September 30, 2023.",
        "What was the total cash paid for interest in fiscal year 2023?",
        "How does the company describe the impact of a weakening of foreign currencies relative to the U.S. dollar on its sales and pricing?",
        "Identify the total amount of cash, cash equivalents, and marketable securities at the end of fiscal year 2023.",
        "As of September 30, 2023, what were the total future principal payments for term debt maturing after 2028?",
        "What was the total share-based compensation expense for 2023?",
        "According to the footnotes, what two vendors represented a significant portion of total vendor non-trade receivables in 2023, and what was their combined percentage?",
        "What are the primary components of 'Other Non-Current Assets' for 2023?",
        "How does the company manage interest rate risk on its term debt?",  # additional stuff
        # --- Direct Data Retrieval (Granular & Varied Phrasing) ---
        "What is the Commission File Number for Apple Inc.?",  # Meta
        "What is Apple's state of incorporation?",  # Meta
        "Find the zip code for Apple's principal executive offices.",  # Meta
        "What was the total number of shareholders of record as of October 20, 2023?",
        "Identify the total cost of sales for Services in 2022.",
        "What was the value of 'Total current assets' at the end of fiscal year 2022?",
        "Find the amount of cash generated by operating activities in 2021.",
        "How much was listed for 'Accounts payable' on the 2023 Consolidated Balance Sheet?",
        "What were the 'Total operating expenses' for the fiscal year 2022?",
        "Retrieve the 'Basic' earnings per share for 2022.",
        "What was the value of 'Total non-current liabilities' for the fiscal year ended September 30, 2023?",
        "Find the amount of 'Accumulated deficit' as of September 30, 2023.",
        "What was the total amount of dividends declared per share in fiscal year 2022?",
        "How much cash was paid for income taxes, net, in 2022?",
        "What was the value of 'Property, plant and equipment, net' in 2022?",
        "Identify the total gross value of 'Machinery, equipment and internal-use software' in 2023.",
        "What was the interest and dividend income for 2023?", "Find the total interest expense for 2021.",
        "What was the foreign provision for income taxes in 2023?",
        "How much were the foreign pretax earnings in 2022?", "What was the statutory federal income tax rate in 2021?",
        "Find the value of 'Deferred tax assets' within 'Other Non-Current Assets' for 2023.",
        "What was the total amount of gross unrecognized tax benefits as of September 24, 2022?",
        "What was the total of 'Net sales' for the Americas segment in 2021?",
        "Find the 'Operating income' for the Japan segment in 2021.", "What were the net sales for the U.S. in 2022?",
        "How much was the value of long-lived assets in China as of September 24, 2022?",
        "What was the total 'Share-based compensation expense' recognized in 2021?",
        "How much in 'Total lease liabilities' was recorded for 2022?",
        "What was the weighted-average interest rate of the Company's commercial paper as of September 24, 2022?",
        "Find the total term debt principal as of September 30, 2023.",
        "How many shares of common stock were outstanding at the end of fiscal year 2021?",
        "What was the aggregate fair value of RSUs canceled in 2022?",
        "Identify the total unrecognized compensation cost related to outstanding RSUs as of September 30, 2023.",
        "What is the total value of future payments for unconditional purchase obligations for the year 2025?",
        "Find the total operating income for the Greater China segment in 2023.",
        "Who is listed as the Senior Director of Corporate Accounting and Principal Accounting Officer?",  # Meta
        "What was the net income for the fiscal year ended September 25, 2021?",
        "How much revenue was recognized in 2023 that was included in deferred revenue as of the end of the 2022 fiscal year?",
        "What was the total number of RSUs outstanding as of September 24, 2022?",
        "Find the total amount of 'Term debt' in current liabilities for 2022.",

        # --- Comparative & Trend Analysis ---
        "Compare the total cost of sales in 2023 versus 2021.",
        "What was the percentage change in net sales for the Europe segment between 2022 and 2023?",
        "Analyze the trend in operating income from 2021 to 2023.",
        "How did the gross margin on Products change from 2022 to 2023?",
        "What was the difference in total assets between fiscal year-end 2023 and 2022?",
        "Compare the amount of cash used in financing activities in 2023 versus 2021.",
        "Did the company's investment in property, plant, and equipment increase or decrease from 2022 to 2023?",
        "What was the trend in the company's total number of employees from 2022 to 2023 (if available)?",
        # Note: Only 2023 is listed
        "Compare the effective tax rate in 2023 to the rate in 2021.",
        "How did the value of total deferred revenue change from September 24, 2022, to September 30, 2023?",
        "What was the change in 'Vendor non-trade receivables' from 2022 to 2023?",
        "Compare the net sales of the Rest of Asia Pacific segment in 2023 and 2021.",
        "Did the operating income for the Americas segment grow from 2021 to 2023? By how much?",
        "What has been the trend of 'Total shareholders' equity' over the three years shown in the report?",
        "Compare the cash paid for interest in 2023 versus 2022.",
        "How have 'Total current liabilities' changed from the end of FY2022 to the end of FY2023?",
        "Analyze the year-over-year change in R&D expenses as a percentage of total net sales from 2021 to 2023.",
        "What was the difference in net income between 2023 and 2021?",
        "Compare the proceeds from the issuance of term debt in 2023 versus 2021.",
        "How did the number of diluted shares used in computing EPS change from 2022 to 2023?",
        "What was the trend for iPad net sales from 2021 to 2023?",
        "Compare the Services gross margin percentage in 2023 with the percentage in 2021.",
        "How did the amount of cash and cash equivalents at the end of the year change between 2021 and 2023?",
        "What was the change in total long-lived assets in the U.S. from 2022 to 2023?",
        "Compare the number of RSUs granted in 2023 versus 2022.",
        "How did the amount of commercial paper outstanding change from the end of FY2022 to FY2023?",
        "What was the difference in total operating expenses between 2023 and 2021?",
        "Compare the operating income of the Europe segment in 2023 with its income in 2021.",
        "Analyze the change in the 'Accumulated other comprehensive loss' from the start to the end of the three-year period shown.",
        "How did the payments for taxes related to net share settlement of equity awards trend from 2021 to 2023?",

        # --- Narrative & Explanatory Retrieval ---
        "What does the 'Business' section identify as the Company's five main product categories?",
        "According to the 'Competition' section, how do many of Apple's competitors seek to compete?",
        "What are the principal competitive factors important to the Company, as listed in the report?",
        "Describe the risks associated with the Company's reliance on single or limited source suppliers for components.",
        "How does the report characterize the historical seasonality of the business?",
        "Summarize the company's policies on workplace inclusion and diversity.",
        "What are the main categories of services Apple provides, as described under the 'Services' heading?",
        "How does the company define the 'Rest of Asia Pacific' segment?",
        "What does the report state about the impact of adverse macroeconomic conditions on consumer spending?",
        "Describe the risks related to political events and trade disputes mentioned in the 'Risk Factors' section.",
        "According to the report, why must the company successfully manage frequent introductions of new products?",
        "What does the document say about the potential for design and manufacturing defects in its complex products?",
        "Explain the risk associated with write-downs on the value of inventory.",
        "How does the report describe the company's dependence on third-party software developers?",
        "What risks related to the App Store are discussed, particularly concerning commissions and legal challenges?",
        "What does the report say about the challenge of creating and licensing digital content?",
        "Explain the importance of key personnel to the company's success, according to the 'Risk Factors'.",
        "Describe the risks associated with the performance of carriers, wholesalers, and other resellers.",
        "What potential consequences are mentioned regarding losses or unauthorized access to confidential information?",
        "Summarize the risks associated with investing in new business strategies and acquisitions.",
        "What factors pose risks and uncertainties to the company's retail store operations?",
        "Describe the status of the 'Epic Games' legal proceeding as of the report date.",
        "What does the report state about the potential outcomes of unfavorable legal proceedings or government investigations?",
        "How do complex and changing laws worldwide expose the company to potential liabilities?",
        "What does the report say about the increasing focus on environmental, social, and governance (ESG) considerations?",
        "Explain the risks to the company's business related to its licensing arrangements for search services.",
        "Describe the types of data protection laws the company is subject to globally.",
        "According to the MD&A, what factors can cause the company's gross margins to be volatile?",
        "What explanation is given in the MD&A for the 4% decrease in Americas net sales in 2023?",
        "What reasons are provided for the increase in Services net sales during 2023?",
        "Why did the Products gross margin percentage increase during 2023 compared to 2022?",
        "What was the primary driver of the growth in R&D expense in 2023?",
        "What does the 'Critical Accounting Estimates' section say about 'Uncertain Tax Positions'?",
        "How does the company manage foreign exchange rate risk, according to the 'Quantitative and Qualitative Disclosures About Market Risk' section?",
        "What is the company's policy for recognizing revenue from the sale of its products?",
        "How does the company account for arrangements with multiple performance obligations, like the sale of an iPhone?",
        "What is the purpose of the Apple Inc. 2022 Employee Stock Plan?",
        "Where are the majority of the company's hardware products manufactured, according to Note 12?",
        "What is the basis for how the company evaluates the performance of its reportable segments?",
        "What is the primary responsibility of management as stated in the 'Report of Independent Registered Public Accounting Firm'?",
        # Meta

        # --- Deeper, Component-Based & Inferential Questions ---
        "Calculate Apple's gross margin percentage for fiscal year 2023.",
        "What percentage of total net sales did the iPhone represent in 2023?",
        "Calculate the ratio of total liabilities to total assets for fiscal year 2023.",
        "What was the total amount of debt (Commercial Paper + Term Debt) maturing within one year as of September 30, 2023?",
        "Based on the table in the MD&A, which geographic segment had the highest percentage decrease in net sales in 2023?",
        "What was the total amount spent on share repurchases and dividends in fiscal year 2023, according to the Statement of Cash Flows?",
        "From the 'CONSOLIDATED STATEMENTS OF SHAREHOLDERS' EQUITY', what was the beginning balance of Retained Earnings for fiscal year 2022?",
        "What percentage of total 'Other Non-Current Assets' did 'Deferred tax assets' represent in 2023?",
        "In Note 7, what were the two largest components of 'Total deferred tax assets' before valuation allowance in 2023?",
        "What is the total of future lease payments for operating leases for the years 2026, 2027, and 2028 combined?",
        "What amount of term debt matures in 2026?",
        "Calculate the total value of RSUs that vested during fiscal year 2023.",
        "What is the total value of unconditional purchase obligations with a term from 2026 to 2028?",
        "What was the total of 'Other corporate expenses, net' used in the reconciliation of segment operating income for 2023?",
        "Calculate the operating margin for the Europe segment in 2023 (Operating Income / Net Sales).",
        "What percentage of long-lived assets were located in the U.S. as of September 30, 2023?",
        "The report mentions an 'accelerated share repurchase agreement' (ASR) in August 2023. What was the up-front payment amount?",
        "How is revenue from product-related bundled services and unspecified software upgrade rights recognized over time?",
        "What was the total income tax benefit related to share-based compensation expense in 2023?",
        "How many RSUs were excluded from the computation of diluted EPS for 2023 because their effect would have been antidilutive?",
        "What is the maximum length of time over which the Company is hedging its exposure to variability in future cash flows for term debt transactions?",
        "As of September 24, 2022, what percentage of total trade receivables did the company's third-party cellular network carriers account for?",
        "What was the total depreciation expense on property, plant and equipment in 2022?",
        "What were the three components of 'Other income/(expense), net' for fiscal year 2023?",
        "In the reconciliation of the provision for income taxes, what was the value attributed to 'Earnings of foreign subsidiaries' in 2023?",
        "What was the adjusted recovery amount for the European Commission State Aid Decision as of September 30, 2023?",
        "What was the discount rate related to the Company's lease liabilities as of September 30, 2023?",
        "What was the weighted-average grant date fair value per RSU for RSUs granted in 2023?",
        "What is the total amount of future principal payments for the Company's Notes for the period 'Thereafter' (after 2028)?",
        "If the company has a minority market share in smartphones, PCs, and tablets, what can be inferred about the competitive landscape?",
        "What was the net effect of changes in operating assets and liabilities on the cash generated by operating activities in 2023? (Sum of the relevant lines)",
        "Calculate the percentage of total net sales that came from the Americas segment in 2023.",
        "What was the total value of 'Marketable securities' (current and non-current) on the balance sheet for September 30, 2023?",
        "What was the net change in cash from investing activities in 2022?",
        "What was the value of 'Common stock issued' in fiscal year 2023, according to the Statement of Shareholders' Equity?",
        "What was the total current federal provision for income taxes in 2023?",
        "Calculate the total value of non-current marketable debt securities due after 5 years as of September 30, 2023.",
        "What was the net impact of 'Change in unrealized gains/losses on marketable debt securities' on 'Other comprehensive income/(loss)' in 2022?",
        "What percentage of total operating expenses was 'Selling, general and administrative' in 2023?",
        "Which product category saw the largest percentage decrease in net sales from 2022 to 2023?",

        # --- More Granular/Complex Questions ---
        "What was the total of 'Unrealized Losses' on Level 2 Corporate debt securities as of September 30, 2023?",
        "What was the fair value of non-current marketable securities that were U.S. agency securities in 2023?",
        "How much in 'Imputed interest' was subtracted from the total undiscounted liabilities for Finance Leases?",
        "For the debt issuance in Q3 2023, what is the maturity year for the 4.850% notes?",
        "Who signed the Power of Attorney in the SIGNATURES section of the report?",  # Meta
        "What is the title of Exhibit 10.18 as listed in the Exhibit Index?",  # Meta
        "According to the MD&A, what new subscription service was announced in the first quarter of 2023?",
        "In the legal proceedings for the Masimo case, what specific Apple Watch models are mentioned as being affected by the ITC's order?",
        "In the share repurchase activity table for Q4 2023, what was the average price paid per share for purchases between July 2, 2023, and August 5, 2023?",
        "What was the carrying amount of hedged current and non-current term debt in 2022?",
        "As of September 24, 2022, what two vendors accounted for 54% and 13% of total vendor non-trade receivables, respectively?",
        "In the reconciliation of segment operating income to total operating income, what types of costs are included in 'Other corporate expenses, net'?",
        "Since when has Ernst & Young LLP served as Apple Inc.'s auditor?",  # Meta
        "What is the total amount of 'Proceeds from/(Repayments of) commercial paper, net' for maturities 90 days or less in 2022?",
        "What was the change in 'Unrealized gains' on Level 2 U.S. Treasury securities from 2022 to 2023?",
        # Requires checking both tables
        "What does the report describe as 'visionOS'?",
        "What specific risk is mentioned regarding ransomware attacks in the cybersecurity risk factor section?",
        "In the 'Critical Audit Matter' section of the auditor's report, what specific account is identified as being challenging to audit?",
        "How does the company determine stand-alone selling prices (SSPs) when observable prices are not available?",
        "What was the fair value as of the respective vesting dates for RSUs in 2021?",
        "What was the total dollar value of the share repurchase program announced on May 4, 2023?",
        "What was the Company's quarterly cash dividend per share after it was raised in May 2023?",
        "According to the stock performance graph, approximately what was the value of a $100 investment in the Dow Jones U.S. Technology Supersector Index as of September 2021?",
        "What are the three performance obligations regularly included in arrangements involving the sale of an iPhone?",
        "What was the value of 'Tax credit carryforwards' in deferred tax assets for 2022?",
        "What was the amount of future payments for noncancelable unconditional purchase obligations for 'Thereafter' (after 2028)?",
        "What were the net sales for the 'Mac' product category in 2021?",
        "How did the gross value of 'Leasehold improvements' change from 2022 to 2023?",
        "What was the 'Total shareholders' equity' at the end of fiscal year 2021?",
        "Find the cash flow adjustment for 'Changes in operating assets and liabilities: Other current and non-current assets' in 2023.",
        "What was the total of 'Common stock withheld related to net share settlement of equity awards' against 'Retained earnings' in 2022?",
        "What was the net change in unrealized gains/losses on derivative instruments in 'Other comprehensive income/(loss)' for 2023?",
        "Calculate the percentage of total current assets that 'Cash and cash equivalents' represented at the end of fiscal year 2023.",
        "What was the total operating income for the 'Rest of Asia Pacific' segment in 2022?",
        "How much revenue was recognized in 2022 that was part of deferred revenue from the end of fiscal year 2021?",
        "What was the weighted-average period over which the company expects to recognize the total unrecognized compensation cost for RSUs?",
        "What was the total fair value of derivative assets designated as accounting hedges as of September 24, 2022?",
        "What was the value of 'Long-term taxes payable' in 'Other Non-Current Liabilities' for 2023?",
        "In the tax reconciliation table, what was the value of the 'Research and development credit, net' in 2022?",
        "What was the total amount of gross unrecognized tax benefits that, if recognized, would impact the effective tax rate as of September 30, 2023?",
        "What was the total of undiscounted liabilities for operating leases as of September 30, 2023?",
        "What was the effective interest rate on the 2013-2022 debt issuances as of September 24, 2022?",
        "How many shares of common stock were repurchased in 2022?",
        "What was the weighted-average grant date fair value per RSU for all RSUs outstanding as of September 25, 2021?",
        "What is the total value of future payments under noncancelable unconditional purchase obligations as of September 30, 2023?",
        "What was the value of net sales attributed to 'Other countries' (excluding U.S. and China) in 2021?",
        "According to the report, on what date did Deirdre O'Brien enter into a trading plan under Rule 10b5-1(c)?",
        "What was the primary reason for the decrease in Europe's net sales in 2023?",
        "What were the total proceeds from maturities of marketable securities in 2023?",
        "What was the value of the adjustment for net gains/losses on marketable debt securities realized and included in net income for 2023?"]


if __name__ == "__main__":
    langfuse_handler = setup_langfuse()
    run_config = RunnableConfig(callbacks=[langfuse_handler])
    vector_store = initialize_vector_store()
    rag_test_questions = get_rag_test_questions()

    print("Starting query execution...")
    for i, question in enumerate(rag_test_questions):
        print(f"\n--- Running Query {i + 1}/{len(rag_test_questions)} ---")
        print(f"Question: {question}")
        results = execute_query(question, vector_store, 10, run_config)
        pprint(results)
    print("\n--- Query execution complete. ---")
