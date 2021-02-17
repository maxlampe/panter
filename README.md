![image](panter_logo.png)

# panter

panter (Perkeo ANalysis Tool for Evaluation and Reduction) is a Python package for reducing, evaluating and analyzing PERKEO III data.

## Project status

This project is very much WIP and changes are constantly being made. Feel free to contact me for feedback and issues.
Currently, core features are still implemented and can be subject to change.

## Installation

Get the package from gitlab: https://gitlab.lrz.de/ge39dat/panter

Python version 3.8 or above required.

```bash
git clone https://gitlab.lrz.de/ge39dat/panter.git
pip3 install -r requirements.txt
```

## Usage

See docstrings and python help() function. For example:

```python
#Core data structure
from panter.core.dataPerkeo import RootPerkeo
help(RootPerkeo)
#Data loader
from panter.core.dataloaderPerkeo import DLPerkeo
help(DLPerkeo)
#General Fitter and specific fitting analysis
from panter.core.evalPerkeo import DoFit, DoFitData
help(DoFit)
help(DoFitData)
#Data reduction and correction class
from panter.core.corrPerkeo import corrPerkeo
help(corrPerkeo)
```
etc.

## Unit tests

All unit tests can be found in the _tests_ directory. Run _alltests.py_ for all of them.

## License
[MIT](https://choosealicense.com/licenses/mit/)
