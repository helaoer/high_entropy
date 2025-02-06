Source code for "Co-expression of Multi-genes for Polynary Perovskite Electrocatalysts"

Use the command:

    conda env create -f environment_skopt.yml

to install all required packages.

train.py is an example script for training BOSRs. You can find the corresponding scripts in the rp750, rp700, and rp650 folders. You can also find scripts for model ensembling in these folders if needed.

equation_converter.ipynb converts the generated plain equation into plain text. It is recommended to use LLMs to convert the text into LaTeX equations rather than directly using the generated equation.

heatmap.ipynb provides an example for feature correlation analysis and mRFE.

Each folder contains the corresponding scripts and data.

If you encounter the error:

    TypeError: got an unexpected keyword argument 'squared'

while running the BOSRs, modify the following line:

    mean_squared_error(y, y_pred, squared=False)

to:

    root_mean_squared_error(y, y_pred)

This is because mean_squared_error no longer supports the squared argument in newer versions of scikit-learn. Remember to import the root_mean_squared_error function.

MC-MD and AIMD folder contains the raw data for molecular dynamics simulations.