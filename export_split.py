"""
Exports the train/dev/test split of a training config for a dataset converted with csv_to_hdf5.py.
Writes split.csv (split of each event) and the CSV files of the selected split in the original CSV format.
"""
import argparse
import csv
import json
import os
import pandas as pd

import loader

SPLIT_PARTS = {'train': (True, False, False),
               'dev': (False, True, False),
               'test': (False, False, True)}


def get_split_events(config):
    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])[0]
    data_path = training_params['data_path']
    if isinstance(data_path, list):
        if len(data_path) > 1:
            raise NotImplementedError('Exporting splits for joint training is not supported')
        data_path = data_path[0]

    split_events = {}
    for split, parts in SPLIT_PARTS.items():
        event_metadata, _, _ = loader.load_events(data_path, parts=parts,
                                                  shuffle_train_dev=generator_params.get('shuffle_train_dev', False),
                                                  custom_split=generator_params.get('custom_split', None),
                                                  min_mag=generator_params.get('min_mag', None),
                                                  mag_key=generator_params.get('key', 'MA'),
                                                  decimate_events=generator_params.get('decimate_events', None),
                                                  data_keys=[])
        split_events[split] = event_metadata
    return split_events


def filter_csv_by_event(input_file, output_file, event_idxs):
    # Filters line by line to keep the original formatting of the values
    event_idxs = {str(x) for x in event_idxs}
    with open(input_file, 'r', encoding='utf-8', newline='') as f_in, \
            open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        header = f_in.readline()
        event_col = next(csv.reader([header])).index('event_idx')
        f_out.write(header)
        for line in f_in:
            if '"' in line:
                fields = next(csv.reader([line]))
            else:
                fields = line.split(',', event_col + 1)
            if fields[event_col].strip() in event_idxs:
                f_out.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)  # Training config defining data path and split
    parser.add_argument('--input', type=str, required=True)  # Folder containing the CSV files
    parser.add_argument('--output', type=str, required=True)  # Output folder
    parser.add_argument('--split', type=str, default='test', choices=list(SPLIT_PARTS.keys()))
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    split_events = get_split_events(config)

    os.makedirs(args.output, exist_ok=True)
    split = pd.concat([pd.DataFrame({'event_idx': events['event_idx'].values,
                                     'event_id': events['#EventID'].values,
                                     'split': name})
                       for name, events in split_events.items()])
    split.to_csv(os.path.join(args.output, 'split.csv'), index=False)

    event_idxs = split_events[args.split]['event_idx'].values
    for name in ['event_metadata', 'stations', 'waveforms']:
        filter_csv_by_event(os.path.join(args.input, f'{name}.csv'),
                            os.path.join(args.output, f'{name}_{args.split}.csv'),
                            event_idxs)

    print(', '.join(f'{name}: {len(events)} events' for name, events in split_events.items()))
    print(f'Exported {args.split} split to {args.output}')
