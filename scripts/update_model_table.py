import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from l2m2.model_info import MODEL_INFO, PROVIDER_INFO

header = "| Model Name | Provider(s) | Model Version(s) |\n"
sep = "| --- | --- | --- |\n"
rows = ""


def get_provider_link(provider_key):
    provider_name = PROVIDER_INFO[provider_key]["name"]
    provider_homepage = PROVIDER_INFO[provider_key]["homepage"]
    return f"[{provider_name}]({provider_homepage})"


def make_row(model_name):
    providers = []
    model_ids = []
    for provider_key, details in MODEL_INFO[model_name].items():
        providers.append(get_provider_link(provider_key))
        model_ids.append(f"`{details['model_id']}`")
    return f"| `{model_name}` | {', '.join(providers)} | {', '.join(model_ids)} |\n"


for model_name in MODEL_INFO:
    rows += make_row(model_name)

markdown_table = "\n\n" + header + sep + rows + "\n"

readme_path = "../README.md"
table_start = "<!--start-model-table-->"
table_end = "<!--end-model-table-->"
count_start = "<!--start-count-->"
count_end = "<!--end-count-->"

with open(readme_path, "r") as f:
    readme = f.read()
    new_readme = readme.replace(
        readme[readme.index(table_start) : readme.index(table_end) + len(table_end)],
        f"{table_start}{markdown_table}{table_end}",
    ).replace(
        readme[readme.index(count_start) : readme.index(count_end) + len(count_end)],
        f"{count_start}{len(MODEL_INFO)}{count_end}",
    )


with open(readme_path, "w") as f:
    f.write(new_readme)

print("Updated model table in README.md")
