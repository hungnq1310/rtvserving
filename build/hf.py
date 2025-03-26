import os
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from jinja2 import Template


HF_CONFIG_FILE = os.getenv("HF_CONFIG_FILE", "/hf.json")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "/models")
HF_MODEL_REF = os.getenv("HF_MODEL_REF", "main")

def build_template(repos: str, tokens: str) -> str:
  """
  tokens: str - A comma separated string of tokens
  repos: str - A comma separated string of repos
  """
  # prepare
  template_str = """
    {
      "models": [
        {% for model in models %}
        {
          "name": "{{ model.name }}",
          "ref": "{{ model.ref }}",
          "token": "{{ model.token }}"
        }{% if not loop.last %},{% endif %}
        {% endfor %}
      ]
    }
    """
  template = Template(template_str)
  tokens=tokens.split(",") # list[str]
  repos=repos.split(",") # list[str]

  # check if tokens and repos are the same length
  if len(tokens) == 1:
    tokens = [tokens[0] for _ in range(len(repos))]
  else:
      assert len(repos) == len(tokens), "Invalid number of tokens and repos!"
  #  create data from tokens and repos
  data_list = []
  for repo, token in zip(repos, tokens):
      data_list.append({"name": repo, "ref": HF_MODEL_REF, "token": token})
  data = {
      "models": data_list
  }
  # render
  rendered_json = template.render(data)
  return rendered_json


if __name__ == "__main__":

  file = Path(HF_CONFIG_FILE).expanduser().resolve()
  model_repo = Path(HF_MODEL_REPO)
  model_repo.mkdir(parents=True, exist_ok=True)

  conf = None
  if not file.is_file():
    print("No huggingface config found!")
    # Build jinja template from githubaction
    repos = os.getenv("HF_REPOS", None)
    tokens = os.getenv("HF_TOKENS", None)
    conf = json.loads(build_template(repos, tokens))
  else:
    with file.open("r") as f:
      conf = json.load(f)
  # download models
  if not conf:
    print("Error create config!")
    exit(1)
  token = conf.get("token", None)
  models = conf.get("models", [])
  for model in models:
    _name = model.get("name", None)
    assert _name is not None, "Invalid huggingface config! Model name cannot be none!"
    _token = model.get("token", token)
    _ref = model.get("ref", None)
    snapshot_download(repo_id=_name, revision=_ref, token=_token, local_dir=model_repo, ignore_patterns=[".*"])
  cache = Path(model_repo, ".cache")
  shutil.rmtree(cache)
