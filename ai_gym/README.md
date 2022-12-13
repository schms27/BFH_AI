Creat a cliff hanger env and train it with monte-carlo

Start the `venv`:
```
$ source venv/bin/activate
```

Export the required pip installs:
```
pip3 freeze > requirements.txt
```

OR use a conda env alternatively:
```
conda env create -f .\environment.yaml
```

Update an existing env after changes on environment.yaml:
```
conda activate pytorch-rl
conda env update -f .\environment.yaml --prune
```