"""
Example script demonstrating how to call and run Xanthos.
"""
import os
import warnings
warnings.filterwarnings("ignore")

from xanthos import Xanthos


def run(ini):

    # instantiate model
    xth = Xanthos(ini)

    # run intended configuration
    xth.execute()

    return xth


if __name__ == "__main__":

    # full path to parameterized config file
    ini = os.path.join('/xanthos-wm/pm_abcd_mrtm_managed.ini')

    # run the model
    xth = run(ini)

    del xth