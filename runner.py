import argparse
import csv
import requests
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
from pathlib import Path

import datalad.api as api
from datalad.api import Dataset, create, copy_file, remove, clone, get
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

studies = ["CCNP", "BHRC", "NKI", "HBN", "PNC"]
git_repos = [f"https://github.com/ReproBrainChart/{s}_CPAC.git" for s in studies]

### Study Parameters ###
default_dict = dict(
    task="rest",
    run=None,
    acq=None,
    atlas="CC200",
    space="MNI152NLin6ASym",
    reg="36Parameter",
)
study_parameters = {study: default_dict.copy() for study in studies}
study_parameters["PNC"]["acq"] = "singleband"
study_parameters["NKI"]["acq"] = "645"

for s in ["CCNP", "BHRC", "HBN"]:
    if s == "CCNP":
        run = "01"
    else:
        run = "1"
    study_parameters[s]["run"] = run

### Acquisition Times in ms ###
acquisition_times = dict(
    CCNP=2.5,
    BHRC=2.000,
    NKI=0.645,
    HBN=dict(SI=1.450, CBIC=0.800, RU=0.800, CUNY=0.8),
    PNC=3.000,
)


def compute_dynamic_connectome(fpath, window_length=60, tr=None):
    """
    Parameters
    ----------
    window_length : int
        Length of window to compute dynamic connectome in seconds. Default=45.
    tr : int
        TR in seconds
    """
    assert tr is not None

    df = pd.read_csv(fpath)
    tr_per_window = int(np.round(window_length / tr))

    corrs = []
    for i in range(len(df) - tr_per_window):
        corr = np.corrcoef(df[i : i + tr_per_window], rowvar=False)
        idx = np.triu_indices_from(corr)
        corrs.append(corr[idx])

    return np.array(corrs)

def main(args):
    out_path = Path(args.output)

    # Grab the metadata files
    for study in studies:
        study_path = out_path / f"{study}"
        if not study_path.exists():
            study_path.mkdir(exist_ok=True)
    
        url = f"https://raw.githubusercontent.com/ReproBrainChart/{study}_BIDS/main/study-{study}_desc-participants.tsv"
        response = requests.get(url)
        reader = csv.reader(response.text.splitlines(), skipinitialspace=True)
    
        with open(study_path / f"{study}_desc-participants.tsv", "w") as f:
            w = csv.writer(f)
            w.writerows(reader)

    # Datalad clone the datasets
    for git_repo in git_repos:
        api.clone(source=git_repo, git_clone_opts=["-b", "complete-pass-0.1"])
    
    for study, study_parameter in study_parameters.items():
        # load metadata
        df = pd.read_csv(out_path / f"{study}/{study}_desc-participants.tsv", delimiter="\t")
        df = df[~df["p_factor_mcelroy_harmonized_all_samples"].isnull()]
    
        print(f"Computing dynamic connectomes for {study}; Total files={len(df)}")
    
        # Setup glob string
        glob_str = "*".join(
            [f"{k}-{v}" for k, v in study_parameter.items() if v is not None]
        )
        glob_str = "**/*" + glob_str + "*Mean_timeseries.1D"
    
        p = Path(f"./{study}_CPAC/cpac_RBCv0")

        files = []
        # Loop over each row of metadata
        for _, row in df.iterrows():
            sub_path = p / f"sub-{row.participant_id}/ses-{row.session_id}"
    
            # print
            tmp = list(sub_path.glob(glob_str))
            
            if len(tmp) == 0:
                continue
            else:
                files += tmp
    
        # Download all the files
        api.get(files)
    
        for file in files:
            splits = file.name.split("_")
            if study == "HBN":
                site = splits[1][11:]
                tr = acquisition_times[study][site]
            else:
                tr = acquisition_times[study]

            # save file
            participant_id = splits[0]
            sub_path =  out_path / f"{participant_id}"
            sub_path.mkdir(parents=True, exist_ok=True)
            out_fpath = sub_path / file.name

            arr = compute_dynamic_connectome(file, tr=tr)
            np.save(out_fpath, arr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="output path", type=str, default="/output")
    args = parser.parse_args()

    main(args)