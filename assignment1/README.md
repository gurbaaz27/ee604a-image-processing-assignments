# Assignment 1

## Submission

Name: Gurbaaz Singh Nandra

Roll No.: 190349

Date: 28th August 2022

## Setup

Setup a virtual environment to install the following dependencies to work with images

```bash
python -m venv venv
source venv/bin/activate
pip install numpy pillow
```

## Usage

1. `dotmatrix.py`

```bash
python dotmatrix.py --help

usage: dotmatrix.py [-h] n_d

positional arguments:
  n_d         Docking station number nd ∈ {00, 01, ..., 99}

options:
  -h, --help  show this help message and exit
```

e.g.

```bash
python dotmatrix.py 42
```

![](./dotmatrix.jpg)

2. `jigsolver.py`

```bash
python jigsolver.py --help

usage: jigsolver.py [-h] filepath

positional arguments:
  filepath    file path for input jigsaw.jpg

options:
  -h, --help  show this help message and exit
```

e.g.

```bash
python jigsolver.py ./jigsaw.jpg
```

![](./jigsolved.jpg)

3. `cycloneanalyzer.ipynb`

Open the jupyter notebook on [Google Colab](https://colab.research.google.com/) or on local machine's `localhost`

```bash
jupyter notebook cycloneanalyzer.ipynb
```

or simply execute complete code with

```bash
jupyter run cycloneanalyzer.ipynb
```
