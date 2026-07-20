# Configs

This folder is the canonical home for run configuration JSON files. Config
paths are organized to mirror the result namespaces where practical.

## Result Namespace Mapping

- `protocol_1/`: configs for `results/protocol_1`.
  These use the canonical-protocols: `epochs=50`, `batch_size=16`,
  `learning_rate=0.0001`.
- `protocol_2/`: configs for `results/protocol_2`.
  These use the ALIGNN-recommended hyperparameters: `epochs=300`,
  `batch_size=64`, `learning_rate=0.001`.
- `protocol_3/`: configs for `results/protocol_3`.
  These use the low-learning-rate hyperparameters: `epochs=100`,
  `batch_size=32`, `learning_rate=0.00005`.

There should be no flat JSON files directly in this directory. Add durable run
configs to the matching namespace above so config paths stay aligned with result
paths.

## Cache Note

The config JSON files are durable research inputs. The `id_prop.csv*_data/`
LMDB directories created beside training data are generated runtime caches and
should not be committed.
