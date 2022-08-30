import pickle as pl
from pathlib import Path
from os.path import isfile
from itertools import chain
from collections import defaultdict, Counter

import qml
import MDAnalysis
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def read_data():
    # Loading in SLATM representations and concatenating round 0 and round 1
    with open("slatm_round0_21.pickle", 'rb') as f:
        representations0 = pl.load(f)
    with open("slatm_round1_21.pickle", 'rb') as f:
        representations1 = pl.load(f)

    representationsc = representations0 + representations1

    # Loading mbtypes and related data
    with open("mbtypeschargesmappingcompounds.pickle", 'rb') as f:
        mbtypes_small, charges_dict_small, mapping_dict_small, compounds = pl.load(f)

    # Loading selectivites
    selec_round0 = pd.read_pickle("results-round_0.pickle")["ΔΔG PG->CL"]
    selec_round1 = pd.read_pickle("results-round_1.pickle")["ΔΔG PG->CL"]
    col = selec_round0.dropna()
    col_round1 = selec_round1.dropna()
    col_round1.index = 'r1' + col_round1.index
    selectivities = col.append(col_round1)

    # Retrieving all relevant simulation files.
    p = Path("/home/diego/Downloads/Scriptie/Trajectories/")
    gro_list = list(p.glob("**/prod-*.gro"))
    xtc_list = list(p.glob("**/prod-*.xtc"))
    itp_list = list(p.glob("**/molecule_*.itp"))

    