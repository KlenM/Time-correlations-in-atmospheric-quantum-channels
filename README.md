The dataset is contained in the `data` folder.
Each file in the folder corresponds to different aperture radius.
For example, the file `strong_0_02.csv` is the data for the aperture radius equal `0.02` m.
The first row in the file contains the names of the columns with the `wind-driven shift` values (in metres).

To run the code, ensure that you have [Python 3.10+](https://realpython.com/installing-python/#how-to-install-python-on-linux) installed.
Create a [virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments), [install the required packages](https://packaging.python.org/en/latest/tutorials/installing-packages/#use-pip-for-installing) from the `requirements.txt` file:
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Install the `cupy` package of the appropriate version to the installed `cuda` library.
You can check the version of your `cuda` library with the command `nvidia-smi`.
E.g. for the `CUDA Version: 12.3` run `pip install cupy-cuda12x`.

Then start the simulation and plotting with the commands:
```
python simulation.py
python results.py
```
