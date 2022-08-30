import argparse
from pathlib import Path
import pandas as pd
import regex as re


def replace_patterns(replace_dict, template_line):
    rgx = re.compile("|".join(map(re.escape, replace_dict.keys())))
    return rgx.sub(lambda match: replace_dict[match.group(0)], template_line)


def preprocess_batch(dataframe, environments, script):
    # template_line = 'sshpass -p $mypassword scp -r SOURCE_PATH/ROUND/MOLECULE ' \
    #                 'bmohr@carbon.science.uva.nl:/home/bmohr/RERUN_STATE1/LIPID/ROUND/\n'
    template_line = 'sshpass -p $mypassword scp -r ' \
                    'bmohr@carbon.science.uva.nl:/home/bmohr/RERUN_STATE1/LIPID/ROUND/MOLECULE/**/jobid-*/* ' \
                    'SOURCE_PATH/ROUND/MOLECULE/**/ \n'
    replace_dict = {'SOURCE_PATH': None, 'ROUND': None, 'MOLECULE': None, 'LIPID': None}
    cl_path = environments[0]
    pg_path = environments[1]
    cdl2 = cl_path.name
    popg = pg_path.name
    df = pd.read_pickle(dataframe)
    batches = df.groupby('batches')
    for batch in batches:
        name = batch[0]
        print(name)
        batch_copy = script.parent / f'{script.stem.rsplit("_", 1)[0]}_{name}'
        batch_retreive = script.parent / f'retrieve_results-{name}'
        script_text = script.read_text()
        rounds = batch[1].groupby('rounds')
        for rnd in rounds:
            round_ = rnd[0]
            replace_dict['ROUND'] = round_
            molecules = rnd[1]['molecules'].tolist()
            for molecule in molecules:
                replace_dict['MOLECULE'] = molecule
                replace_dict['LIPID'] = cdl2
                replace_dict['SOURCE_PATH'] = str(cl_path)
                next_line = replace_patterns(replace_dict, template_line)
                script_text += next_line
                replace_dict['LIPID'] = popg
                replace_dict['SOURCE_PATH'] = str(pg_path)
                next_line = replace_patterns(replace_dict, template_line)
                script_text += next_line
        # batch_copy.write_text(script_text)
        batch_retreive.write_text(script_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Submit rerun in batches, always all passed systems per molecule')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Pandas dataframe with data in subsamples')
    parser.add_argument('-sys', '--systems', type=Path, nargs='+', required=True, help='All system directories that '
                                                                                       'need to be run.')
    parser.add_argument('-scr', '--script', type=Path, required=True, help='Submit script with parameters for intended '
                                                                           'compute node.')

    args = parser.parse_args()
    preprocess_batch(args.dataframe, args.systems, args.script)