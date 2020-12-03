# panter

panter (Perkeo ANalysis Tool for Evaluation and Reduction) is a Python package for reducing, evaluating and analyzing PERKEO III data.

## Project status

This project is very much WIP and changes are constantly being made. Feel free to contact me for feedback and issues.
Currently, core features are still implemented and can be subject to change.

## Installation

Get the package from gitlab: https://gitlab.lrz.de/ge39dat/panter

```bash
git clone https://gitlab.lrz.de/ge39dat/panter.git
pip3 install -r requirements.txt
```

## Usage

See docstrings and python help() function. For example:

```python
from panter.core.dataPerkeo import RootPerkeo
help(RootPerkeo)
from panter.core.dataloaderPerkeo import DLPerkeo
help(DLPerkeo)
from panter.core.evalPerkeo import DoFit, DoFitData
help(DoFitData)
```
etc.

## License
[MIT](https://choosealicense.com/licenses/mit/)
