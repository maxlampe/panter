
![image](images/panter_logo.png)

# panter

_panter_ (**P**erkeo **AN**alysis **T**ool for **E**valuation and **R**eduction) is a Python package for reducing, evaluating and analyzing PERKEO III data.

Created and maintained by [Max Lamparth](https://github.com/maxlampe/)

## Project status

This project is very much WIP and changes are constantly being made. Feel free to contact [me](mailto:max.lamparth@tum.de?subject=panter) for feedback and issues.
Currently, even core features are still implemented and can be subject to change.

## Usage

### Installation

Get the package from [GitHub](https://github.com/maxlampe/panter) and install it locally, as there is no pip release.

Python version 3.8 or above required.

```bash
git clone https://github.com/maxlampe/panter
pip3 install -r requirements.txt
python3 -m pip install -e path-to-panter-dir
```

### Getting started

_panter_ has several core classes. Their relationship is illustrated below in the schematic.

![image](images/schematic.png)

See docstrings and python help() function for manuals and examples. Core classes are listed below.
In _panter/applications/panter_example.py_ additional, but shorter examples can be found as well.

```python
# Core data structure
from panter.data.dataHistPerkeo import HistPerkeo
from panter.data.dataRootPerkeo import RootPerkeo
help(HistPerkeo)
help(RootPerkeo)

# Data loader
from panter.data.dataloaderPerkeo import DLPerkeo
help(DLPerkeo)

# General Fitter and specific fitting analysis
from panter.eval.evalFit import DoFit
help(DoFit)

# Data reduction and correction class
from panter.eval.corrPerkeo import CorrPerkeo
help(CorrPerkeo)
```

### File structure

- Base classes are in _panter/base/_
- Config files, ini files and derived evaluation parameters are in _panter/config/_
- Data classes (e.g. ROOT file import, histogram and the data loader classes) are in _panter/data/_
- Evaluation modules (e.g. fit and data correction classes) are in _panter/eval/_
- Time and spacial map classes are in _panter/maps/_
- Unit tests and their base class are in _tests/_

### Generated files

Generated files from the Perkeo III '19/20 campaign can be found here 
[Google-Drive](https://drive.google.com/drive/folders/1OAMSJ6GS1H43I2-rBBymWHGwYljm34oG?usp=sharing)
and should also be placed into the config directory, should you not generate them yourself.
## Development

### Design decisions

1. Style guide: [pep8](https://www.python.org/dev/peps/pep-0008/)
2. Docstring convention: [numpy](https://numpydoc.readthedocs.io/en/latest/format.html)
3. Ideally, you should use [black](https://pypi.org/project/black/) as code formatter.

**Maximum line length is 88** (default black enforced line length).

### Unit tests

All unit tests can be found in the _tests_ directory. Run _alltests.py_ for all of them.
A working ROOT installation is required to run the unit tests, as the panter and root outputs
are compared.

## License

[MIT](https://choosealicense.com/licenses/mit/)
