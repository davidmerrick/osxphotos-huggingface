import os
import shutil
from pathlib import Path

import click
import torch
import yaml
from huggingface_hub import Repository


@click.command()
@click.option("--token", "-t", required=True)
@click.option("--model_path", "-m", required=True, default="/Users/davidmerrick/code/meme_classifier/model.pth")
@click.option("--config_path", "-c", required=True, default="/Users/davidmerrick/code/meme_classifier/preprocessor_config.json")
@click.option("--repo_id", "-r", required=True, default="davidmerrick/meme_classifier")
def push_pytorch_model_to_hub(model_path, config_path, repo_id, token):
    """
    Push a PyTorch model and its configuration to Hugging Face Hub.

    Args:
        model_path (str): Path to the model file (.pth).
        config_path (str): Path to the config file (.yaml).
        repo_id (str): Hugging Face repository ID (e.g., "username/repo_name").
        token (str): Hugging Face authentication token.
    """
    # Load the model state dictionary
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")

    # Load the configuration
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Define repository directory
    repo_dir = Path("/tmp/temp_hf_repo")

    # Handle non-empty folders or unrelated Git repositories
    if repo_dir.exists():
        if not (repo_dir / ".git").exists():
            print(f"Non-Git folder '{repo_dir}' detected. Removing it.")
            shutil.rmtree(repo_dir)
        else:
            # Check if the existing repo matches the target repo
            os.chdir(repo_dir)
            with os.popen("git remote get-url origin") as remote_check:
                existing_remote = remote_check.read().strip()
            os.chdir("..")
            target_repo_url = f"https://huggingface.co/{repo_id}"
            if existing_remote != target_repo_url:
                print(f"Unrelated Git repository detected at '{repo_dir}'. Removing it.")
                shutil.rmtree(repo_dir)

    # Clone or create the Hugging Face repository
    print(f"Creating or cloning Hugging Face repository: {repo_id}")
    repo_url = f"https://huggingface.co/{repo_id}"
    repo = Repository(local_dir=repo_dir, clone_from=repo_url, use_auth_token=token)

    # Change to the repository directory
    os.chdir(repo_dir)

    # Save the model and config in the repository directory
    print(f"Saving model and config to {repo_dir}...")
    torch.save(state_dict, "model.pth")
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    # Add and commit the changes
    print("Adding and committing changes...")
    repo.git_add(".")
    repo.git_commit("Upload custom PyTorch model and config")

    # Push to the Hugging Face Hub
    print("Pushing to Hugging Face Hub...")
    repo.git_push()
    print(f"Model and config pushed to Hugging Face Hub: {repo_url}")


if __name__ == "__main__":
    push_pytorch_model_to_hub()
