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
    CCNP = 2.5,
    BHRC = 2.000,
    NKI = .645,
    HBN = dict(SI = 1.450, CBIC = .800, RU=.800, CUNY=.8),
    PNC = 3.000
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
        corr = np.corrcoef(df[i:i+tr_per_window], rowvar=False)
        idx = np.triu_indices_from(corr)
        corrs.append(corr[idx])

    return np.array(corrs)

for study in studies:
    out_path = Path(f"./{study}")
    if not out_path.exists():
        out_path.mkdir(exist_ok=True)

    url = f"https://raw.githubusercontent.com/ReproBrainChart/{study}_BIDS/main/study-{study}_desc-participants.tsv"
    response = requests.get(url)
    reader = csv.reader(response.text.splitlines(), skipinitialspace=True)

    with open(out_path / f"{study}_desc-participants.tsv", "w") as f:
        w = csv.writer(f)
        w.writerows(reader)

for study, study_parameter in study_parameters.items():
    # load metadata
    df = pd.read_csv(f"{study}/{study}_desc-participants.tsv", delimiter="\t")
    df = df[~df["p_factor_mcelroy_harmonized_all_samples"].isnull()]

    print(f"Computing dynamic connectomes for {study}; Total files={len(df)}")

    # Setup glob string
    glob_str = "*".join(
        [f"{k}-{v}" for k, v in study_parameter.items() if v is not None]
    )
    glob_str = "**/*" + glob_str + "*Mean_timeseries.1D"

    p = Path(f"./{study}_CPAC/cpac_RBCv0")

    # Loop over each row of metadata
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sub_path = p / f"sub-{row.participant_id}/ses-{row.session_id}"

        # print
        files = list(sub_path.glob(glob_str))
        if len(files) == 0:
            continue

        for file in files:
            if study == 'HBN':
                site = row.session_id[7:]
                tr = acquisition_times[study][site]
            else:
                tr = acquisition_times[study]

            # save file
            out_path = Path(f"./{study}/sub-{row.participant_id}")
            out_path.mkdir(parents=True, exist_ok=True)
            out_fpath = out_path / file.name

            arr = compute_dynamic_connectome(file, tr=tr)
            np.save(out_fpath, arr)