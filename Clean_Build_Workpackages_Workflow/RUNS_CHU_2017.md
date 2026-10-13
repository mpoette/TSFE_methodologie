# Runbook — Runs CHU 2017+ (wp1 & ablation domaines TSFEL)

Documentation des travaux de mi-octobre 2026 sur la pipeline
`Clean_Build_Workpackages_Workflow` (repo `TSFE_methodologie`) :

1. **Phase 3** — artefacts par run dans un `run_YYYYMMDD_HHMMSS/` dédié.
2. **Sous-dataset CHU 2017+** — cohort restreint aux passages `adm_year >= 2017`.
3. **Ablation des domaines TSFEL** — champ config `tsfel_domains` par profil.
4. **Lancements** — commandes tmux, monitoring, post-run.

---

## 1. Phase 3 — un dossier d'artefacts par run

Avant : tous les runs écrivaient dans les arbres partagés
`Clean_Build_Workpackages_Workflow/{inputs,models,outputs}/` + `optuna.db` —
les runs successifs s'accumulaient et ne pouvaient être attribués qu'à la date
(voir `results/README.md` — "arbres fusionnés").

Après : chaque lancement crée **`run_<YYYYMMDD_HHMMSS>[_<config>]/`** dans le
dossier de lancement (CBW) et y écrit **tout** :

```
run_20261012_163456_run_chu_wp1_2017/
├── run_metadata.json        # run_name, run_dir, started_at, config complet
├── inputs/                  # global_brut.parquet, boruta, lasso, variables, ...
├── models/                  # fold_0..4.joblib / .pt par modèle
├── outputs/                 # all_res.joblib, figures, comparisons, holdout
├── optuna.db                # études Optuna du run (plus de db partagée)
└── pipeline_logs/           # pipeline_<stamp>_pid<pid>.log
```

- Le nom du run reprend le `.stem` de `--config` (points remplacés par `_`) :
  `--config configs/run.chu.wp1_2017.json` → `run_<stamp>_run_chu_wp1_2017`.
- La pipeline `chdir` dans le run dir au démarrage (restauré en fin) ;
  `dataset_path` est résolu en absolu s'il est relatif.
- `run_*/` est gitignoré (`.gitignore` CBW) ; les `.out` le sont aussi
  (`.gitignore` racine `*.out`).
- Implémentation : `RunContext` + `Experiment(run_context=..., tsfel_domains=...)`
  dans `utilitaries/path_utils.py` ; branchement dans
  `ICU_Models_Comparison.py` ; `set_storage()` dans
  `utilitaries/optuna/optuna_utils.py` ; tests dans
  `tests/test_run_context.py`.
- **Rétrocompatible** : sans `RunContext`, les chemins restent relatifs au CWD
  comme avant (aucun impact sur un éventuel usage API direct).

Le **run de référence post-Phase-3** : `run_20261012_163456_run_chu_wp1_2017`
(démarré le 12/10 16:34, `.out` : `chu/chu_wp1_2017_20261012_1634.out`).

---

## 2. Sous-dataset CHU 2017+

**Objectif** : n'utiliser que les **passages** (encounters) CHU avec
`adm_year >= 2017` (sémantique confirmée : un patient dont le 1er passage est
2015 mais qui a un passage 2019 conserve le passage 2019).

- **Construction** : `chu/scripts/build_chu_2017plus.py` (polars, ~5 min),
  sortie dans **`/data2/poette.m/chu_2017plus/`** (470 Mo) :
  - `merged_static_ano_and_dynamic.parquet` — filtré `adm_year >= 2017`
    (3 963 305 lignes, 14 760 encounters)
  - `df_static_full_clean.parquet` — filtré `adm_year >= 2017` (42 819 lignes).
    **Indispensable** : la dédup `prio_first` de la pipeline trie ce fichier
    par `utcInTime` et garde le 1er passage par `lifeTimeNumber` ; en le
    filtrant aussi, la dédup opère **parmi les passages ≥ 2017** (et non sur
    l'historique complet, qui exclurait les patients dont le 1er passage est
    antérieur).
  - `df_dynamic_full_clean.parquet` — filtré aux encounterIds retenus
    (non utilisé en mode `Without pmsi`, mais cohérence).
- **Cohort final** (après la dédup `prio_first` intégrée, `keep_duplicates=false`) :
  **12 906 patients** / 3 537 791 lignes merged.
- MIMIC n'a pas de `adm_year` réelle (années dé-identifiées ~2110-2211) :
  le filtre 2017 n'a de sens **que sur CHU**.
- Rappeler : les incohérences historiques de routing (CHU →
  `no_duplicates_prio_first`, MIMIC → `duplicates`) sont déjà portées par
  `keep_duplicates`/`mode_duplicates` dans les configs ; rien à changer ici.

Pour reconstruire le sous-dataset (idempotent) :

```bash
/data2/poette.m/.venvuv/bin/python3.12 chu/scripts/build_chu_2017plus.py
```

---

## 3. Ablation des domaines TSFEL (`tsfel_domains`)

**Contexte** : TSFEL extrait 4 domaines par variable (`statistical`,
`temporal`, `spectral`, `fractal`). La pipeline excluait déjà `fractal`,
`Histogram mode` (stat) et `MFCC` (spectral), soit **59 features TSFEL** par
variable (19 stat + 15 temp + 25 spect).

Étude demandée : comparer une version **statistiques seulement** et une version
**statistiques + temporelles**.

### Utilisation

Nouveau champ **par profil** dans la config JSON (optionnel, vide = comportement
historique) :

```json
"tsfel_domains": ["statistical"]
```

| Valeur                          | Domaines extraits | Features TSFEL/variable |
|---------------------------------|-------------------|-------------------------|
| absent / `[]`                   | stat+temp+spectral (comportement historique) | 59 |
| `["statistical"]`               | statistical        | 19 |
| `["statistical","temporal"]`    | statistical+temporal | 34 |
| `["spectral"]`                  | spectral           | 25 |

Validation : seule `statistical`/`temporal`/`spectral` acceptées.

### Séparation des artefacts

Chaque ensemble de domaines obtient **son sous-arbre d'artefacts** via un slug
dans le chemin de base (`utilitaries/path_utils.py` :
`Experiment.tsfel_domains_slug`) :

```
<run_dir>/inputs/no_duplicates_prio_first/isDeceased_lt_28d/target_col/
  24h_debut_rea_no-fill/Mode_Commonly_Used_Without_pmsi/
  ├── tsfel_statistical/tsfel_global/global_brut.parquet        # variante A
  └── tsfel_statistical_temporal/tsfel_global/global_brut.parquet  # variante B
```

→ pas de partage de cache entre variantes (le cache `global_brut.parquet` est
clé sur le chemin), pas de collision models/outputs. Sans `tsfel_domains`, le
layout est strictement inchangé.

### Code touché

- `utilitaries/run_config.py` — `ModeConfig.tsfel_domains` + validation.
- `utilitaries/features_extraction_utils.py` — `build_tsfel_config(domains=None)`
  (exclusions standing + filtrage par domaines) ; param `domains=` sur
  `extract_tsfel_per_patient`.
- `utilitaries/path_utils.py` — param `tsfel_domains=` sur `Experiment`, slug
  dans `_get_base_path`.
- `ICU_Models_Comparison.py` — `TSFEL_DOMAINS` lu du profil, passé à
  `Experiment` et à l'extraction, affiché dans le banner
  (`TSFEL domains: statistical, temporal`).

Vérifications faites : config 14 exp ; 59/19/34 features ; chemins distincts ;
`py_compile` OK ; `pytest` 6/7 (1 échec pré-existant hors périmètre) ; `ruff`
sans nouveau finding.

---

## 4. Lancements

### 4.1 Run wp1 complet 2017+ — `chu_wp1_2017` (LANCE le 12/10 16:34)

Config : `configs/run.chu.wp1_2017.json` — 1 profil `chu_wp1_2017`,
**7 expériences** (XGBoost / Lasso / RF TSFEL + VanillaTransformer /
InceptionTime / Lstm / SVC TSFEL), optuna ON (Lasso exclu
automatiquement), `run_comparison/run_lasso/run_interpretability/run_test` ON,
`feature_modes = ["Mode Commonly Used Without pmsi"]` uniquement,
`keep_duplicates=false` / `prio_first`.

```bash
# (déjà lancé le 12/10 — relance si nécessaire)
tmux new -s chu2017
cd /home/poette.m/mimic-validation/TSFE_methodologie/Clean_Build_Workpackages_Workflow
STAMP=$(date +%Y%m%d_%H%M)
uv run --extra dev python ICU_Models_Comparison.py \
  --config configs/run.chu.wp1_2017.json \
  > /home/poette.m/mimic-validation/chu/chu_wp1_2017_${STAMP}.out 2>&1
```

- `.out` : `chu/chu_wp1_2017_20261012_1634.out`
- Run dir : `run_20261012_163456_run_chu_wp1_2017/`

### 4.2 Run ablation domaines TSFEL — `chu_tsfeldomains` (À LANCER — demain)

Config : `configs/run.chu.tsfeldomains.json` — **2 profils**, 7 modèles chacun,
= **14 expériences** dans un seul run dir :

| Profil                | `tsfel_domains`           | Sous-arbre d'artefacts |
|-----------------------|---------------------------|------------------------|
| `chu_tsfel_stat`      | `["statistical"]`         | `.../tsfel_statistical/` |
| `chu_tsfel_stat_temp` | `["statistical","temporal"]` | `.../tsfel_statistical_temporal/` |

**Important** : ne lancer qu'**après** le terminaison du run `chu_wp1_2017`
(contention GPU 3090 sur les 3 modèles deep learning).

```bash
tmux new -s tsfeldomains
cd /home/poette.m/mimic-validation/TSFE_methodologie/Clean_Build_Workpackages_Workflow
STAMP=$(date +%Y%m%d_%H%M)
uv run --extra dev python ICU_Models_Comparison.py \
  --config configs/run.chu.tsfeldomains.json \
  > /home/poette.m/mimic-validation/chu/chu_tsfeldomains_${STAMP}.out 2>&1
```

- 14 exp ; les 2 profils s'exécutent en séquentiel dans un même run dir
  `run_<stamp>_run_chu_tsfeldomains/`.
- Estimation : ~2× le temps du run wp1_2017 (2 extractions TSFEL distinctes +
  2×7 entraînements/optuna/lasso/interpretability).

### 4.3 Template générique de config (pour créer une nouvelle variante)

Copier `configs/run.chu.wp1_2017.json`, changer `modes` + le profil. Les seuls
leviers utiles :

```json
{
  "seed": 42,
  "modes": ["mon_mode"],                  // subset de keys de "profiles"
  "run_comparison": true,                  // rapports comparatifs + sous-groupes
  "run_training": true,                    // entraînement 5 folds + calibration
  "run_lasso": true,                       // paths de stabilité lasso
  "run_interpretability": true,            // SHAP / stability v2 (lourd)
  "run_test": true,                        // éval holdout + bootstrap
  "save_figure": true,
  "figure_output_format": "pdf",
  "dataset_path": "/data2/poette.m/chu_2017plus",
  "target_columns": { "Survie à 28 jours": "isDeceased_lt_28d" },
  "keep_duplicates": false,                // CHU = prio_first ; MIMIC = true
  "profiles": {
    "mon_mode": {
      "windowing_mode": "24h début réanimation sans remplissage",
      "target_labels": ["Survie à 28 jours"],
      "model_names": [ "…", "…" ],
      "stratify_modes": ["target_col"],
      "optuna_run_options": [true],         // Lasso exclu automatiquement
      "feature_modes": ["Mode Commonly Used Without pmsi"],
      "balancing_methods": ["Aucune Méthode"],
      "mode_duplicates": "prio_first",
      "tsfel_domains": []                   // optionnel, cf. §3
    }
  }
}
```

Le nombre d'expériences = `target_labels × stratify_modes × feature_modes ×
balancing_methods × (optuna options, 1 pour Lasso)` — vérifier avec
`--print-config` avant de lancer :

```bash
uv run --extra dev python ICU_Models_Comparison.py --config configs/maj.json --print-config
```

### 4.4 Monitoring (tout run)

```bash
# démarrage correct
grep -a "Artifacts will be written to" chu/chu_*.out | tail -1
grep -ac "CRASHED" chu/chu_*.out            # doit rester 0

# suivi
tail -f chu/chu_wp1_2017_20261012_1634.out

# progression
grep -a "EXPERIMENT\|COMPLETED" chu/chu_wp1_2017_20261012_1634.out
```

Le `chdir` dans le run dir rend le terminal silencieux : c'est normal, toute la
sortie est dans le `.out` (équivalent `nohup` non détaché). Ne pas fermer le
tmux sans détacher (`Ctrl-b d`).

### 4.5 Post-run

- Copier config + run dir dans `results/chu/` (convention existante, cf.
  `results/README.md`) — les runs Phase 3 s'y rangent plus proprement que les
  anciens arbres fusionnés.
- Vérifier `run_metadata.json` (config exacte + horodatage) pour citer le run.

---

## Où est stocké quoi (cartographie)

| Artefact | Emplacement |
|---|---|
| **Artèfacts d'un run** (models/, outputs/, inputs/, optuna.db, logs, run_metadata.json) | `TSFE_methodologie/Clean_Build_Workpackages_Workflow/run_<stamp>[_<config>]/` |
| **Sortie console** (.out) | `chu/<nom_run>_<stamp>.out` (à la racine du workspace) |
| **Config du run** | `TSFE_methodologie/Clean_Build_Workpackages_Workflow/configs/run.<nom>.json` (copie d'archive dans `results/chu/configs/`) |
| **Archive de sécurité des runs** | `results/chu/` (copies ; cf. `results/README.md`) |
| **Sous-dataset CHU 2017+** | `/data2/poette.m/chu_2017plus/` (reconstruisible via `chu/scripts/build_chu_2017plus.py`) |
| **Dataset CHU source** | `/data2/paquie.d/Datasets/output/` (non modifié) |
| **Scripts ad hoc** | `chu/scripts/` |
| **Arbres historiques (pré-Phase 3, "fusionnés")** | `chu/`, `TSFE_methodologie/Clean_Build_Workpackages_Workflow/{inputs,models,outputs}` — ne pas nettoyer |

**Reprendre un run** : le run dir contient tout (modèles + résultats +
`run_metadata.json` avec la config exacte). Pour relancer le même run,
relancer la même commande (un nouveau run dir est créé — les artefacts ne sont
jamais écrasés). Pour continuer l'analyse sans retrain, voir les modes
`run_training=false` (ex. `run.chu.dl_reeval.json` dans `chu/configs/`).

---

## 6. État au 12/10 17:30

| Élément | État |
|---|---|
| Phase 3 (run dirs) | Implémentée, validée (tests + smoke CHU 100 patients + run live) |
| Sous-dataset 2017+ | Construit (`/data2/poette.m/chu_2017plus/`, 12 906 patients) |
| `tsfel_domains` | Implémenté, validé (14 exp, 59/19/34 features, chemins distincts) |
| Run `chu_wp1_2017` | **En cours** (7 exp, Boruta en cours au 17:13) |
| Run `chu_tsfeldomains` | Config prêt, **à lancer demain** (après le wp1) |

### À ne pas oublier
- [ ] Lancer `chu_tsfeldomains` demain (après terminaison du wp1) — commande en §4.2.
- [ ] Post-run : copier les 2 runs dans `results/chu/`.
- [ ] Le run `run_20261012_161026_run_smoke_chu100` (validation Phase 3) peut être
      supprimé ; le dataset `/tmp/opencode/chu100` aussi (volatile).
