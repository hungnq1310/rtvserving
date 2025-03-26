import os
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from jinja2 import Template


HF_CONFIG_FILE = os.getenv("HF_CONFIG_FILE", "/hf.json")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "/models")

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
      ],
      "token": "{{ token }}"
    }
    """
  template = Template(template_str)
  tokens=tokens.split(",") # list[str]
  repos=repos.split(",") # list[str]

  #  create data from tokens and repos
  if len(tokens) == 1:
      data_list = []
      for repo in repos:
          data_list.append({"name": repo, "ref": "main", "token": ""})
      data = {
          "models": data_list,
          "token": tokens[0] # general token for all models
      }
  elif len(tokens) == len(repos):
      data_list = []
      for repo, token in zip(repos, tokens):
          data_list.append({"name": repo, "ref": "main", "token": token})
      data = {
          "models": data_list,
          "token": ""
      }
  else:
     raise ValueError("Invalid number of tokens and repos! Number of tokens must be equal to number of repos or 1")
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
    repos = os.getenv("REPOS", None)
    tokens = os.getenv("TOKENS", None)
    conf = json.loads(build_template(repos, tokens))
    print("Generated config: ", conf)
    print("repos: ", repos)
    print("tokens: ", tokens)
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
