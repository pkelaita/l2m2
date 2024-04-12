import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from l2m2 import llm_client

header = "| Provider | Model Name | Model Version |\n"
sep = "| --------- | ---------- | ------------- |\n"
rows = ""

for model_name, details in llm_client._MODEL_INFO.items():
    provider_link = f"[`{details['provider']}`]({details['provider_homepage']})"
    rows += f"| {provider_link} | `{model_name}` | `{details['model_id']}` |\n"

markdown_table = header + sep + rows
print(markdown_table)
