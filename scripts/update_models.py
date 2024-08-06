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

model_table = "\n\n" + header + sep + rows + "\n"

json_native = "\n"
for model_name in MODEL_INFO:
    for provider in MODEL_INFO[model_name]:
        extras = MODEL_INFO[model_name][provider].get("extras", {})
        if extras.get("json_mode_arg", None) is not None:
            json_native += f"\n- `{model_name}` (via {provider.capitalize()})"
json_native += "\n\n"

readme_path = "../README.md"
count_start = "<!--start-count-->"
count_end = "<!--end-count-->"
table_start = "<!--start-model-table-->"
table_end = "<!--end-model-table-->"
json_native_start = "<!--start-json-native-->"
json_native_end = "<!--end-json-native-->"


def replace_between(full_string, start, end, replacement):
    try:
        start_idx = full_string.index(start) + len(start)
        end_idx = full_string.index(end)
        return full_string[:start_idx] + replacement + full_string[end_idx:]
    except ValueError:
        print(f"Could not find {start} or {end} in the string)")
        return full_string


with open(readme_path, "r") as f:
    out = f.read()

out = replace_between(out, count_start, count_end, str(len(MODEL_INFO)))
out = replace_between(out, table_start, table_end, model_table)
out = replace_between(out, json_native_start, json_native_end, json_native)

with open(readme_path, "w") as f:
    f.write(out)

print("Updated model table in README.md")
