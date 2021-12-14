import json
import os.path
import numpy as np
import pandas as pd
import tqdm

# This script ingests the raw individual level responses from the
# Global survey on COVID-19 beliefs, behaviors (Collis et al, 2020).
# It codes string responses as numeric to be used in the analyses.
# The file maps.json contains the codebook.

path = 'data/covid_survey_responses.txt.gz'
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/covid_survey_responses_numeric.txt.gz')
if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath))
df = pd.read_table(path, sep='\t', low_memory=False)

with open('maps.json', 'r') as f:
    colmap = json.load(f)['numeric_map']

null_vals = [-1, -2, '-1', '-2', '-1.0', '-2.0', np.nan]
split_stubs = [el.replace('__split__', '') for el in colmap if '__split__' in el]
new_df = pd.DataFrame(index=df.index, columns=[])
for col in tqdm.tqdm(df.columns):
    if col in colmap:
        val_map = colmap[col]
    else:
        stub = ''
        stub_length = -1
        for el in split_stubs:
            if col.startswith(el) and len(el) > stub_length:
                stub = el
                stub_length = len(el)
        if stub + '__split__' not in colmap:
            continue
        val_map = colmap[stub + '__split__']

    new_df[col] = np.nan
    if isinstance(val_map, type(None)) or (isinstance(val_map, str) and val_map.lower() == 'none'):
        new_df[col] = df[col]
        new_df.loc[new_df[col].isin(null_vals), col] = np.nan
    else:
        possible_vals = list(val_map.keys()) + null_vals
        for el in df[col].unique():
            assert el in possible_vals, el
            if el in null_vals:
                continue
            else:
                val = val_map[el]
                new_df.loc[df[col] == el, col] = val

new_df.to_csv(outpath, sep='\t', index=False)
