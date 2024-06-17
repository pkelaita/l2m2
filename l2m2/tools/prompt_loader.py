import re


class PromptLoader:
    def __init__(
        self,
        prompts_base_dir: str = ".",
        var_open: str = "{{",
        var_close: str = "}}",
    ) -> None:
        """Initializes the prompt loader.

        Args:
            prompts_base_dir (str, optional): The base directory to load prompts from. Defaults to the current
                directory.
            var_open (str, optional): The opening marker for prompt variables. Defaults to "{{".
            var_close (str, optional): The closing marker for prompt variables. Defaults to "}}".
        """

        self.prompts_base_dir = prompts_base_dir
        self.var_open = var_open
        self.var_close = var_close

    def load_prompt_str(self, prompt: str, variables: dict[str, str] = {}) -> str:
        """Loads a prompt from a string and replaces variables with values.

        Args:
            prompt (str): The prompt string to load.
            variables (dict, optional): A dictionary of variables to replace in the prompt. Defaults to {}.

        Returns:
            str: The loaded prompt with variables replaced.

        Raises:
            ValueError: If a variable is denoted in the prompt but not provided in the variables dictionary.
        """

        vlist = re.findall(
            f"{re.escape(self.var_open)}(.*?){re.escape(self.var_close)}",
            prompt,
        )

        for var in vlist:
            if var not in variables:
                raise ValueError(f"Variable '{var}' not provided in variables.")

            prompt = prompt.replace(
                f"{self.var_open}{var}{self.var_close}", variables[var]
            )

        return prompt

    def load_prompt(self, prompt_file: str, variables: dict[str, str] = {}) -> str:
        """Loads a prompt from a file and replaces variables with values.

        Args:
            prompt_file (str): The name of the prompt file to load.
            variables (dict, optional): A dictionary of variables to replace in the prompt. Defaults to {}.

        Returns:
            str: The loaded prompt with variables replaced.

        Raises:
            ValueError: If a variable is denoted in the prompt but not provided in the variables dictionary.
        """

        prompt_path = f"{self.prompts_base_dir}/{prompt_file}"
        with open(prompt_path, "r") as f:
            prompt = f.read()

        return self.load_prompt_str(prompt, variables)
