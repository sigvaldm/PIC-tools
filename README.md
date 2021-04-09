# PIC tools

In a standard workflow each project is given a project folder (e.g. `Chamber`),
and within this folder there's a folder for each simulation run. The naming
convention of the run folders is specific to each project, but as a rule it is
convenient to encode values of parameters being varied in the name. E.g., if
the density and temperature varies, a folder could be named `Step1_10n_1500K`
to indicate a density of `10e10` and a temperature of `1500K`. This will allow
parsing of the folder name for batch post-processing of all folders. I
sometimes use a prefix such as `Step1` to indicate that this is the first
attempt at getting all the parameters right, and so forth.

I typically store everything related to a given simulation within its run
folder: input files, output files, post-processed results, etc. That makes it
easy to backtrack to exactly which input parameters were used for a given
result. The files have standardized names, such that post-processing scripts
only need the path to the folder as an argument. The scripts can read from
multiple files, and save new results into the folder with standardized name.
One may run consecutive scripts if necessary. The scripts are typically not
general, but customized to each project.

### disprel_split.py
``` bash
usage: disprel.py [-h] [-i INPUT] [-per] [-yLoc YLOCATION] [-pl] [-n NORM]

Plasma Dispersion Processor

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to E-field data
  -per, --periodic      Add this if the system is periodic in Y
  -yLoc YLOCATION, --yLocation YLOCATION
                        In bounded (in Y) system Choose Y location, Options: e.g. 1 (Any number between 0-Ny)
  -pl, --plot           Add this if you want to plot the figure
  -n NORM, --norm NORM  Normalizing frequency, Options: omega_pi, omega_pe)
```
#### Example
``` bash
./disprel.py -i <path_to_data> -per -pl -n omega_pe
```
