"""
Example script demonstrating how to call and run Xanthos.
"""
import os
import sys
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
    ini = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'pm_abcd_mrtm_managed.ini'
    )

    # run the model
    xth = run(ini)

    del xth
