# Dataset card

## Source and target

The study uses the JARVIS-DFT `dft_3d_2021` collection and predicts formation energy per atom. JARVIS identifiers are the stable join keys across family manifests, fixed-test predictions, run splits, and embedding metadata.

## Operational family definitions

- **Oxide comparator:** every structure containing oxygen; structures containing both O and N remain in this arm.
- **Nitride target:** every structure containing nitrogen and no oxygen.

The definitions are intentionally asymmetric and retain 499 oxynitrides in the oxide arm. They are operational comparison groups, not formal statistical distribution labels.

| Family | All | Train | Validation | Fixed test | Pool |
|---|---:|---:|---:|---:|---:|
| Oxide | 14,991 | 11,960 | 1,547 | 1,484 | 13,507 |
| Nitride | 2,288 | 1,837 | 209 | 242 | 2,046 |

## What is tracked

- `manifests/`: official global split mapping and conflict audit;
- `{oxide,nitride}/manifests/`: family membership and train/validation/test/pool membership;
- `{oxide,nitride}/summaries/summary.json`: definitions, source, and counts;
- `diagnostics/`: deduplicated catalog and split/schema checks;
- `{oxide,nitride}/alignn_ready/`: lightweight manifests for ALIGNN input roots.

Individual structure files are not committed because they dominated the former repository and were duplicated across runs. Rebuild them deterministically with:

```bash
python scripts/setup/rebuild_family_datasets.py
```

The wrapper requires `data/manifests/dft_3d_formation_energy_peratom_splits.csv`, writes structures from JARVIS-DFT, and validates family membership, disjoint splits, union identities, and the expected counts above.

## Data provenance and terms

See `PROVENANCE.md` for the archived commit pin and upstream sources. JARVIS/NIST data remain governed by their original terms; this repository does not relicense upstream records.

