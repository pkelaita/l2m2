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
start_marker = "<!--start-model-table-->"
end_marker = "<!--end-model-table-->"

with open(readme_path, "r") as f:
    readme = f.read()
    new_readme = readme.replace(
        readme[readme.index(start_marker) : readme.index(end_marker) + len(end_marker)],
        start_marker + markdown_table + end_marker,
    )

with open(readme_path, "w") as f:
    f.write(new_readme)
    print(f"Updated {readme_path} with model table")
