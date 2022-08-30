import argparse
from pathlib import Path
import pandas as pd
import subprocess
from subprocess import PIPE,STDOUT


def preprocess_batch(dataframe, batch, environments, script):
    scriptpath = Path()
    scriptpath = scriptpath.cwd()  # SCRIPTDIR
    script_text = script.read_text()
    df = pd.read_pickle(dataframe)
    df = df[df['batches'] == f'batch_{str(batch)}']
    cdl2 = sorted(environments[0].glob('ROUND_*'))
    popg = sorted(environments[1].glob('ROUND_*'))
    for cdl2, popg in zip(cdl2, popg):
        rnd = cdl2.name
        cl = cdl2.parts[-2]
        pg = popg.parts[-2]
        mols = df.loc[df['rounds'] == rnd, 'molecules'].tolist()
        cdl2 = [path for mol in mols for path in sorted(cdl2.glob('molecule_*')) if mol in path.name]
        popg = [path for mol in mols for path in sorted(popg.glob('molecule_*')) if mol in path.name]
        for mol_cl, mol_pg in zip(cdl2, popg):
            mol = mol_cl.parts[-1]
            mol_cl = [path for path in list(mol_cl.glob('*')) if path.is_dir()][0].resolve()  # WORKDIR
            mol_pg = [path for path in list(mol_pg.glob('*')) if path.is_dir()][0].resolve()  # WORKDIR
            number = mol_cl.name.split('-')[-1]  # NMBR
            cl_path = script.parent / f'{rnd}-{cl}-{mol}-equ{script.suffix}'
            cl_text = script_text
            cl_text = cl_text.replace('NMBR', number)
            cl_text = cl_text.replace('WORKDIR', str(mol_cl))
            cl_text = cl_text.replace('SCRIPTDIR', '/media/bmohr/Backup/STRUCTURAL_ANALYSIS')
            cl_path.write_text(cl_text)
            # subprocess.call(['bash', cl_path], shell=True, stdout=PIPE,stderr=STDOUT)
            pg_path = script.parent / f'{rnd}-{pg}-{mol}-equ{script.suffix}'
            pg_text = script_text
            pg_text = pg_text.replace('NMBR', number)
            pg_text = pg_text.replace('WORKDIR', str(mol_pg))
            pg_text = pg_text.replace('SCRIPTDIR', '/media/bmohr/Backup/STRUCTURAL_ANALYSIS')
            pg_path.write_text(pg_text)
            # subprocess.call(['bash', pg_path], shell=True, stdout=PIPE,stderr=STDOUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Submit rerun in batches, always all passed systems per molecule')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Pandas dataframe with data in subsamples')
    parser.add_argument('-sys', '--systems', type=Path, nargs='+', required=True, help='All system directories that '
                                                                                       'need to be run.')
    parser.add_argument('-b', '--batch', type=int, required=True, help='Number of the subset to be submitted.')
    parser.add_argument('-scr', '--script', type=Path, required=True, help='Submit script with parameters for intended '
                                                                           'compute node.')

    args = parser.parse_args()
    preprocess_batch(args.dataframe, args.batch, args.systems, args.script)
