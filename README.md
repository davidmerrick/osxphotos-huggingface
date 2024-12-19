# Harmonia: An AI-powered Apple Photos organizer

Harmonia is a set of scripts for organizing your photos with AI models, 
and for fine-tuning your own AI models on your photos.

Shoutout to [osxphotos](https://github.com/RhetTbull/osxphotos), which is heavily used in this project.

![](img/harmonia.jpg)

_In Greek mythology, Harmonia is the goddess of harmony, balance, and peace. And organizing photos._

# Installing dependencies

```shell
# Make sure numpy is installed via brew
brew install numpy

# Pop into a venv
python3 -m venv venv; source venv/bin/activate.fish
pip3 install --upgrade pip setuptools wheel numpy cython
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install -r requirements.txt --verbose
```

# Setup with IntelliJ

1. Open the project in IntelliJ
2. Go to settings and add an existing Python interpreter (venv/bin/python)
3. Set the `PYTHONPATH` environment variable to the root of the project so it can resolve the `lib` module.
4. You might need to add a package prefix of `lib` to your package settings under Project Settings -> Modules. That way your imports will work in the IDE.

# Flagging your photos with AI classifiers

Edit the `./bin/flag_multi.py` script to enable or disable the classifiers you want. Then run:

```shell
PYTHONPATH=$(pwd) ./venv/bin/osxphotos run ./bin/flag_multi.py
```

# Putting photos in separate albums

First, set up your albums in your config file. The default is `~/.config/harmonia/config.yaml`.

An example config is included under `example.config.yml`.

```shell
PYTHONPATH=$(pwd) ./venv/bin/osxphotos run ./bin/add_flagged_to_albums.py
```

# Tuning models

First, create albums for training in Photos. For example, "memes" and "not memes" albums.

Next, set up your training config in your config file. The default is `~/.config/harmonia/config.yaml`.

An example config is included under `example.config.yml`.

Then run:

```shell
PYTHONPATH=$(pwd) ./venv/bin/osxphotos run ./bin/train_models.py
```

# Pushing to huggingface

I'm storing my torch models in huggingface.
To push new models, use the `bin/push_model_to_huggingface.py` script.

# Todo

* Figure out how to package this so it runs as a daemon, for running nightly flagging jobs.
