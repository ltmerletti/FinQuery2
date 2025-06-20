import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from document_ingestion_agent.get_json_from_report import get_json_from_reports

load_dotenv()

client = genai.Client(api_key=(os.getenv("GOOGLE_AI_STUDIO_API_KEY")))

script_dir = Path(__file__).parent

media = script_dir / "reports"

jsons = get_json_from_reports(["aapl-20230930.pdf", "nflx-20231231.pdf", "pltr-20231231.pdf"], media, client)
# json = get_json_from_report("aapl-20230930.pdf", media, client)

print(jsons)
# print(json)
