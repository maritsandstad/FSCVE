# FSCVE - Forest Sensitive Climate Variable Emulator

FSCVE the Forest Sensitive Climate Variable Emulator developed for the Project PATHFINDER.
PATHFINDER has received funding from the European Union’s Horizon Europe Program 
(Climate, Energy, and Mobility) under grant agreement number 101056907. 
This output reflects only the authors’ view, and the European Union cannot be held 
responsible for any use that may be made of the information contained herein.


## Code

Code is structured as a library, main code in src/fscve
- `fscve.py` is the main module which can be fed predictor datasets to produce predictions
- `ml_modelling_infrastructure.py` has infrastructure and a class to hold what is needed from a machine learning model, with training and predictions. The main module uses this to make predictions
- `data_readying_library.py`, `forest_data_handler.py`, and `get_data_from_era5land.py` all provide infrastructure to
read in import and structure data so it can be used in the other modules

The code should be runnable everywhere, however, several of the code parts need some paths for input data which will need to be edited for the code to work outside of ciceros servers. This will be updated in a later version

## Scripts
scripts folder includes a demonstration script that shows the workflow of making, training and making predictions

## Developement
The folder also includes structure for more streamlined developement. 

The Makefile will enable users to set up a pip environment to get libraries needed to run the code.
To get this working do: 
> <code>make first-venv</code>

> <code>make clean</code>

> <code>make virtual-environment</code>

This should only be necessary the first time you setup the code

You can load this environment with
> <code>source venv/bin/activate</code>

Later to update you should do:
> <code>make virtual-environment</code>

Or if you know you need updates, but aren't getting them:
> <code>make clean</code>

> <code>make virtual-environment</code>

After this you should be able to run the automatic tests
> <code>make test</code>

Will only run the tests
> <code>make checks</code>

Will run the tests and formatting tests

Before your code branch can be merged into the main code, it should pass all the tests
(The makefile also has an option to run only the formatting checks ` make format-checks`)
Tests are located in tests divided between unit and integration tests.