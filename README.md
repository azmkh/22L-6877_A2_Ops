# Task 1.4 toy: Git + DVC on a tiny CSV

Separate history from `main` (VisionGuard). `data.csv` is tracked by DVC; Git stores only
`data.csv.dvc`. Two versions: `Data v1: two rows` and `Data v2: three rows`. The data is in the
DagsHub storage of `azmkh/22L-6877_A2_Ops`.

```bash
git clone -b dvc-toy https://github.com/azmkh/22L-6877_A2_Ops.git dvc-toy && cd dvc-toy
dvc pull                                         # version 2 (3 rows)
git checkout HEAD~3 -- data.csv.dvc && dvc pull  # version 1 (2 rows)
```
Pulling needs DagsHub credentials (`dvc remote modify --local dagshub access_key_id <token>`, same for `secret_access_key`).
