"""
Exports a trained model as a TensorFlow SavedModel.

Signature 'serving_default':
    Inputs:
        waveforms: (batch, max_stations, trace_length, 3)
        coords: (batch, max_stations, 3) station latitude, longitude, depth
    Outputs:
        magnitude: (batch, magnitude_mixture, 3) Gaussian mixture as (alpha, mu, sigma)
        location: (batch, location_mixture, 7) Gaussian mixture as (alpha, mu lat, mu lon, mu depth,
                  sigma lat, sigma lon, sigma depth) in units of 100 km relative to pos_offset
"""
import argparse
import json
import os
import tensorflow as tf
import keras.backend as K

import models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--weight_file', type=str)  # If not given, uses the last event weight file
    parser.add_argument('--output', type=str, required=True)  # SavedModel folder, must not exist
    parser.add_argument('--trace_length', type=int, default=3000)  # Samples per trace used in training
    args = parser.parse_args()

    config = json.load(open(os.path.join(args.experiment_path, 'config.json'), 'r'))
    model_params = config['model_params']
    if config.get('ensemble', 1) > 1:
        raise NotImplementedError('Exporting ensembles is not supported. Export each member separately.')
    if model_params.get('n_pga_targets', 0) or model_params.get('dataset_bias', False) \
            or model_params.get('no_event_token', False):
        raise NotImplementedError('Only magnitude and location models are supported')

    K.set_learning_phase(0)
    _, model = models.build_transformer_model(**model_params, trace_length=args.trace_length)

    if args.weight_file is not None:
        weight_file = os.path.join(args.experiment_path, args.weight_file)
    else:
        weight_file = sorted([x for x in os.listdir(args.experiment_path) if x[:5] == 'event'])[-1]
        weight_file = os.path.join(args.experiment_path, weight_file)
    print(f'Loading weights from {weight_file}')
    model.load_weights(weight_file)

    tf.saved_model.simple_save(K.get_session(), args.output,
                               inputs={'waveforms': model.inputs[0], 'coords': model.inputs[1]},
                               outputs={'magnitude': model.outputs[0], 'location': model.outputs[1]})
    print(f'Exported SavedModel to {args.output}')
