# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # One Asset HANK Model [<cite data-cite="6202365/ECL3ZAR7"></cite>](https://cepr.org/active/publications/discussion_papers/dp.php?dpno=13071)
#
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/HARK/BayerLuetticke/notebooks?filepath=HARK%2FBayerLuetticke%2FOneAsset-HANK.ipynb)
#
# This notebook solves a New Keynesian model in which there is only a single liquid asset.  This is the second model described in [Bayer and Luetticke (2019)](https://cepr.org/active/publications/discussion_papers/dp.php?dpno=13071)

# %% {"code_folding": [], "tags": [], "jupyter": {"source_hidden": true}}
# Setup stuff 

# This is a jupytext paired notebook that autogenerates a corresponding .py file
# which can be executed from a terminal command line via "ipython [name].py"
# But a terminal does not permit inline figures, so we need to test jupyter vs terminal
# Google "how can I check if code is executed in the ipython notebook"

def in_ipynb():
    try:
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            return True
        else:
            return False
    except NameError:
        return False

# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline') 
else:
    from matplotlib.pyplot import ion
    ion()
    get_ipython().run_line_magic('matplotlib', 'auto') 

# The tools for navigating the filesystem
import sys
import os
import pickle
import time
import warnings
from copy import copy

# Ignore scary but unimportant system warnings while running the notebook
warnings.filterwarnings('ignore')

# %% {"code_folding": [], "tags": [], "jupyter": {"source_hidden": true}}
# Code must be inside a main() block to be usable for multiprocessing from command line
# Jupyter notebooks ignore the multiprocessing (so are slower)
def main():

    # Find pathname to this file:

    my_file_path = os.path.dirname(os.path.abspath("OneAsset-HANK.py"))
    
    # Relative and absolute paths for pickled code
    code_dir_rel = os.path.join(my_file_path, "../Assets/One") 
    code_dir = os.path.abspath(code_dir_rel)
    sys.path.insert(0, code_dir)
    sys.path.insert(0, my_file_path)
    os.chdir(code_dir)
    
    from FluctuationsOneAssetIOUsBond import FluctuationsOneAssetIOUs, SGU_solver, plot_IRF

    ## Load precomputed Stationary Equilibrium (StE) object
    # EX1SS.p is the information in the stationary equilibrium
    EX2SS = pickle.load(open("EX2SS.p", "rb"))
    start_time = time.perf_counter() 
    
    # Uncertainty Shock
    
    EX2SS['par']['aggrshock'] = 'Uncertainty'
    EX2SS['par']['rhoS'] = 0.84    # Persistence of variance
    EX2SS['par']['sigmaS'] = 0.54    # STD of variance shocks

    EX2SR=FluctuationsOneAssetIOUs(**EX2SS)
    SR=EX2SR.StateReduc()

    SGUresult=SGU_solver(SR['Xss'],SR['Yss'],SR['Gamma_state'],SR['Gamma_control'],SR['InvGamma'],SR['Copula'],
                         SR['par'],SR['mpar'],SR['grid'],SR['targets'],SR['P_H'],SR['aggrshock'],SR['oc'])

    plot_IRF(SR['mpar'],SR['par'],SGUresult['gx'],SGUresult['hx'],SR['joint_distr'],
             SR['Gamma_state'],SR['grid'],SR['targets'],SR['os'],SR['oc'],SR['Output'])

    # Monetary Policy Shock

    EX2SS['par']['aggrshock'] = 'MP'
    EX2SS['par']['rhoS'] = 0.0      # Persistence of variance
    EX2SS['par']['sigmaS'] = 0.001    # STD of variance shocks

    EX2SR=FluctuationsOneAssetIOUs(**EX2SS)
    SR=EX2SR.StateReduc()

    SGUresult=SGU_solver(SR['Xss'],SR['Yss'],SR['Gamma_state'],SR['Gamma_control'],SR['InvGamma'],SR['Copula'],
                         SR['par'],SR['mpar'],SR['grid'],SR['targets'],SR['P_H'],SR['aggrshock'],SR['oc'])

    plot_IRF(SR['mpar'],SR['par'],SGUresult['gx'],SGUresult['hx'],SR['joint_distr'],
             SR['Gamma_state'],SR['grid'],SR['targets'],SR['os'],SR['oc'],SR['Output'])
    
    # Productivity Shock

    EX2SS['par']['aggrshock'] = 'TFP'
    EX2SS['par']['rhoS'] = 0.95
    EX2SS['par']['sigmaS'] = 0.0075

    EX2SR=FluctuationsOneAssetIOUs(**EX2SS)
    SR=EX2SR.StateReduc()

    SGUresult = SGU_solver(SR['Xss'],SR['Yss'],SR['Gamma_state'],SR['Gamma_control'],SR['InvGamma'],SR['Copula'],
                         SR['par'],SR['mpar'],SR['grid'],SR['targets'],SR['P_H'],SR['aggrshock'],SR['oc'])

    plot_IRF(SR['mpar'],SR['par'],SGUresult['gx'],SGUresult['hx'],SR['joint_distr'],
             SR['Gamma_state'],SR['grid'],SR['targets'],SR['os'],SR['oc'],SR['Output'])
    
if __name__ == "__main__":
    main()
