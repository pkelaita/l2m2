# L2M2: A Simple Python LLM Manager 💬👍

[![CI](https://github.com/pkelaita/l2m2/actions/workflows/ci.yml/badge.svg?timestamp=1763339431)](https://github.com/pkelaita/l2m2/actions/workflows/ci.yml) [![codecov](https://codecov.io/github/pkelaita/l2m2/graph/badge.svg?token=UWIB0L9PR8)](https://codecov.io/github/pkelaita/l2m2) [![PyPI version](https://badge.fury.io/py/l2m2.svg?timestamp=1763339431)](https://badge.fury.io/py/l2m2)

**L2M2** ("LLM Manager" &rarr; "LLMM" &rarr; "L2M2") is a tiny and very simple LLM manager for Python that exposes lots of models through a unified API.

![](docs/assets/l2m2-demo.gif)

### Advantages

- **Simple:** Completely unified interface – just swap out the model name.
- **Tiny:** Only one external dependency (aiohttp). No BS dependency graph.
- **Private:** Compatible with self-hosted models on your own infrastructure.
- **Fast**: Fully asynchronous and non-blocking if concurrent calls are needed.

### Features

- 70+ regularly updated supported models from popular hosted providers.
- Support for self-hosted models via [Ollama](https://ollama.ai/).
- Manageable chat memory – even across multiple models or with concurrent memory streams.
- JSON mode
- Prompt loading tools

### Supported API-based Models

L2M2 supports <!--start-model-count-->71<!--end-model-count--> models from <!--start-prov-list-->[OpenAI](https://openai.com/api/), [Google](https://ai.google.dev/), [Anthropic](https://www.anthropic.com/api), [Cohere](https://docs.cohere.com/), [Mistral](https://docs.mistral.ai/deployment/laplateforme/overview/), [Groq](https://wow.groq.com/), [Replicate](https://replicate.com/), [Cerebras](https://inference-docs.cerebras.ai), and [Moonshot AI](https://www.moonshot.ai/)<!--end-prov-list-->. The full list of supported models can be found [here](docs/supported_models.md).

## Usage ([Full Docs](docs/usage_guide.md))

### Requirements

- Python >= 3.10
- At least one valid API key for a supported provider, or a working Ollama installation ([their docs](https://github.com/ollama/ollama#readme)).

### Installation

```
pip install l2m2
```

### Environment Setup

If you plan to use an API-based model, at least one of the following environment variables is set in order for L2M2 to automatically activate the provider.

| Provider                | Environment Variable  |
| ----------------------- | --------------------- |
| OpenAI                  | `OPENAI_API_KEY`      |
| Anthropic               | `ANTHROPIC_API_KEY`   |
| Cohere                  | `CO_API_KEY`          |
| Google                  | `GOOGLE_API_KEY`      |
| Groq                    | `GROQ_API_KEY`        |
| Replicate               | `REPLICATE_API_TOKEN` |
| Mistral (La Plateforme) | `MISTRAL_API_KEY`     |
| Cerebras                | `CEREBRAS_API_KEY`    |
| Moonshot AI             | `MOONSHOT_API_KEY`    |

Otherwise, ensure Ollama is running – by default L2M2 looks for it at `http://localhost:11434`, but this can be configured.

### Basic Usage

```python
from l2m2.client import LLMClient

client = LLMClient()

response = client.call(model="gpt-5", prompt="Hello world")
print(response)
```

For the full usage guide, including memory, asynchronous usage, local models, JSON mode, and more, see [Usage Guide](docs/usage_guide.md).

## Planned Features

- Streaming responses
- Support for AWS Bedrock, Azure OpenAI, and Google Vertex APIs.
- Support for structured outputs where available (OpenAI, Google, Cohere, Groq, Mistral, Cerebras)
- Response format customization: i.e., token use, cost, etc.
- Support other self-hosted providers (vLLM and GPT4all) outside of Ollama
- Support for batch APIs where available (OpenAI, Anthropic, Google, Groq, Mistral)
- Support for embeddings as well as inference
- Port this project over to TypeScript
- ...etc.

## Contributing

Contributions are welcome! Please see the below contribution guide.

- **Requirements**
  - Python versions 3.10 through 3.14
  - [uv](https://docs.astral.sh/uv/getting-started/installation/) >= 0.9.2
  - [GNU Make](https://www.gnu.org/software/make/)
- **Setup**
  - Clone this repository and create a Python virtual environment.
  - Install dependencies: `make init`.
  - Create a feature branch and an [issue](https://github.com/pkelaita/l2m2/issues) with a description of the feature or bug fix.
- **Develop**
  - Run lint, typecheck and tests: `make` (`make lint`, `make type`, and `make test` can also be run individually).
  - Generate test coverage: `make coverage`.
  - If you've updated the supported models, run `make update-docs` to reflect those changes in the README.
  - Make sure to run `make tox` regularly to backtest your changes back to 10.0 (you'll need to have all versions of Python between 3.10 and 3.14 installed to do this locally. If you don't, this project's CI will still be able to backtest on all of these versions once you push your changes).
- **Integration Test**
  - Create a `.env` file at the project root with your API keys for all of the supported providers (`OPENAI_API_KEY`, etc.).
  - Integration test your local changes by running `make itl` ("integration test local").
  - Once your changes are ready to build, run `make build` (make sure you uninstall any existing distributions).
  - Run the integration tests against the distribution with `make itest`.
- **Contribute**
  - Create a PR and ping me for a review.
  - Merge!

## Contact

If you have requests, suggestions, or any other questions about l2m2 please shoot me a note at [pierce@kelaita.com](mailto:pierce@kelaita.com), open an issue on [Github](https://github.com/pkelaita/l2m2/issues), or DM me on [Slack](https://join.slack.com/t/genai-collective/shared_invite/zt-285qq7joi-~bqHwFZcNtqntoRmGirAfQ).
