These are a few different scripts for fine-tuning huggingface classifiers on your Apple Photos library,
and for running classifiers. The executables are all under the `bin` directory, and the library files are under `lib`.

# Installing dependencies

I found that on my Mac, I couldn't just install the dependencies from requirements.txt and had to take a few manual
steps first. Here are those:

```shell
# Make sure numpy is installed via brew
brew install numpy

# Pop into a venv
python3 -m venv venv; source venv/bin/activate.fish
pip3 install --upgrade pip setuptools wheel numpy cython
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install -r requirements.txt --verbose
```

# Pushing to huggingface

I'm storing my torch models in huggingface. 
To push new models, use the `bin/push_model_to_huggingface.py` script.

# Setup with IntelliJ

1. Open the project in IntelliJ
2. Go to settings and add an existing Python interpreter (venv/bin/python)
3. Set the `PYTHONPATH` environment variable to the root of the project so it can resolve the `lib` module.
4. You might need to add a package prefix of `lib` to your package settings under Project Settings -> Modules. That way your imports will work in the IDE.

# Running scripts:

```shell
PYTHONPATH=$(pwd) ./venv/bin/osxphotos run ./bin/flag_multi.py
```
