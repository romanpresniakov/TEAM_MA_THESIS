"""
Converts a dataset exported as CSV files (manifest.csv, event_metadata.csv, stations.csv, waveforms.csv)
into the HDF5 layout read by loader.load_events.
"""
import argparse
import numpy as np
import pandas as pd
import h5py


def convert(input_path, output_path):
    manifest = pd.read_csv(f'{input_path}/manifest.csv')
    events = pd.read_csv(f'{input_path}/event_metadata.csv')
    stations = pd.read_csv(f'{input_path}/stations.csv')

    sampling_rates = manifest['sampling_rate_hz'].unique()
    if len(sampling_rates) != 1:
        raise ValueError(f'All events need the same sampling rate. Found: {sampling_rates}')
    num_samples = manifest['num_samples'].unique()
    if len(num_samples) != 1:
        raise ValueError(f'All events need the same number of samples. Found: {num_samples}')
    num_samples = int(num_samples[0])

    waveforms = pd.read_csv(f'{input_path}/waveforms.csv',
                            usecols=['event_idx', 'station_idx', 'sample_idx', 'n', 'e', 'z'],
                            dtype={'event_idx': np.int32, 'station_idx': np.int32, 'sample_idx': np.int32,
                                   'n': np.float32, 'e': np.float32, 'z': np.float32})

    # The event order determines the temporal train/dev/test split in the loader
    events = events.sort_values('Timestamp').reset_index(drop=True)
    events.to_hdf(output_path, key='metadata/event_metadata', mode='w', format='table')

    with h5py.File(output_path, 'a') as f:
        f.create_dataset('metadata/sampling_rate', data=int(sampling_rates[0]))
        g_data = f.create_group('data')
        for _, event in events.iterrows():
            event_idx = event['event_idx']
            event_stations = stations[stations['event_idx'] == event_idx].sort_values('station_idx')
            event_waveforms = waveforms[waveforms['event_idx'] == event_idx]

            data = np.zeros((len(event_stations), num_samples, 3), dtype=np.float32)
            data[event_waveforms['station_idx'].values, event_waveforms['sample_idx'].values] = \
                event_waveforms[['n', 'e', 'z']].values

            g_event = g_data.create_group(str(event['#EventID']))
            g_event.create_dataset('waveforms', data=data, compression='lzf')
            g_event.create_dataset('coords', data=event_stations[['latitude', 'longitude', 'depth_km']].values)
            g_event.create_dataset('p_picks', data=event_stations['p_pick_sample'].values.astype(np.int64))
            g_event.create_dataset('pga', data=event_stations['pga'].values)
            g_event.create_dataset('pgv', data=event_stations['pgv'].values)

    print(f'Wrote {len(events)} events to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)  # Folder containing the CSV files
    parser.add_argument('--output', type=str, required=True)  # HDF5 output path
    args = parser.parse_args()

    convert(args.input, args.output)
