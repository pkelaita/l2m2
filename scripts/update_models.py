import sys
from pathlib import Path

from l2m2.model_info import MODEL_INFO, HOSTED_PROVIDERS

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))


header = "| Model Name | Provider(s) | Model Version(s) |\n"
sep = "| --- | --- | --- |\n"
rows = ""


def get_provider_link(provider_key):
    provider_name = HOSTED_PROVIDERS[provider_key]["name"]
    provider_homepage = HOSTED_PROVIDERS[provider_key]["homepage"]
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


provider_list = ""
for i, provider_key in enumerate(HOSTED_PROVIDERS):
    if i == len(HOSTED_PROVIDERS) - 1:
        provider_list += "and "
    provider_list += f"{get_provider_link(provider_key)}"
    if i < len(HOSTED_PROVIDERS) - 1:
        provider_list += ", "


start_model_count = "<!--start-model-count-->"
start_model_table = "<!--start-model-table-->"
start_json_native = "<!--start-json-native-->"
start_prov_list = "<!--start-prov-list-->"
end_model_count = "<!--end-model-count-->"
end_model_table = "<!--end-model-table-->"
end_json_native = "<!--end-json-native-->"
end_prov_list = "<!--end-prov-list-->"


def replace_between(full_string, start, end, replacement):
    i_s = full_string.find(start)
    while i_s != -1:
        i_e = full_string.find(end, i_s)
        if i_e == -1:
            break
        full_string = (
            full_string[: i_s + len(start)] + str(replacement) + full_string[i_e:]  # type: ignore
        )
        i_s = full_string.find(start, i_e)
    return full_string


readme_path = "../README.md"
with open(readme_path, "r") as f:
    out = f.read()
out = replace_between(out, start_model_count, end_model_count, len(MODEL_INFO))
out = replace_between(out, start_prov_list, end_prov_list, provider_list)
out = replace_between(out, start_json_native, end_json_native, json_native)
with open(readme_path, "w") as f:
    f.write(out)
print("Updated README.md")

supported_models_path = "../docs/supported_models.md"
with open(supported_models_path, "r") as f:
    out = f.read()
    out = replace_between(out, start_model_table, end_model_table, model_table)
with open(supported_models_path, "w") as f:
    f.write(out)
print("Updated supported_models.md")
