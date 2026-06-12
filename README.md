# Qualitative Conflict Analysis Library

A validated Python library for analyzing arbitrary conflict instances using agent attitudes, importance classes, and configurable agreement/opposition semantics.

---

## Table of Contents

- [Overview](#overview)
- [Implementation Notes](#implementation_notes)
- [Installation](#installation)
- [Input Files](#input-files)
  - [attitudes.csv (required)](#1-attitudescsv-required)
  - [agent\_classes.csv (required)](#2-agent_classescsv-required)
  - [issue\_classes.csv (required)](#3-issue_classescsv-required)
  - [agent\_names.csv (optional)](#4-agent_namescsv-optional)
  - [issue\_names.csv (optional)](#5-issue_namescsv-optional)
- [Semantics Reference](#semantics-reference)
  - [Agreement Modes](#agreement-modes)
  - [Opposition Modes](#opposition-modes)
- [CLI Usage](#cli-usage)
  - [List All Coalitions](#list-all-coalitions)
  - [Pareto-Optimal Coalitions under Q\*](#pareto-optimal-coalitions-under-q)
  - [Enumerate Oppositions for a Group](#enumerate-oppositions-for-a-group)
  - [Export Results](#export-results)
- [Complete CLI Reference](#complete-cli-reference)

---

## Overview
This library implements the computational framework introduced in "Bi-Conflicts and Qualitative Dominance in Hierarchical Conflict Analysis" (forthcoming). It provides tools for enumerating and filtering coalition structures, opposition structures, and bi-conflicts from a multi-agent attitude matrix, using Pareto-optimal dominance to identify the most significant conflicts within hierarchically prioritised groups of agents and issues.

The usage of the library is presented using popular Middle East conflict situation. The source code consists of the following files:
- pareto_conf_general.py is the library itself
- middle_east_experiments.ipynb presents how to use this library

Core capabilities:

- **Coalition enumeration** — identify all valid coalitions of agents
- **Opposition analysis** — enumerate opposition relationships for a given group of agents
- **Bi-conflict analysis** — selecting the strongest coalition-oposition structure
- **Pareto frontier filtering** — narrow coalitions/oppositions to those on the Q and Q\* frontier
- **Configurable semantics** — choose how agreement and opposition are defined
- **Friendly name mapping** — attach human-readable labels to agent and issue identifiers
- **Result export** — write outputs to prefixed files for downstream use

---

## Implementation Notes

The model stores agent and issue importance classes alongside a `{-1, 0, 1}` attitude matrix. Each coalition or opposition structure is augmented with integer rank encodings of its agent and issue profiles — mixed-radix numbers that preserve lexicographic profile order — so that dominance comparisons reduce to simple integer comparisons.

All Pareto frontiers avoid the naïve O(n²) pairwise scan. The **Q-frontier** is computed as a 2D skyline on the two rank values in O(n log n) via sort-and-sweep. The **Q\*-frontier** first applies the Q-skyline as a pre-filter (anything eliminated there cannot survive the stricter relation), then runs the full pairwise check only within the small surviving set, costing O(n log n + f²) where f ≪ n in practice. **Bi-conflict frontiers** decompose the problem by computing structure-level frontiers over coalition and opposition components independently, using these to prune candidates, and finishing with a pairwise check on the small remainder — the same two-phase strategy applied to both the Q and Q\* variants.


---

## Installation

No special installation is required beyond a standard Python environment. Clone or copy `pareto_conf_general_csv.py` into your working directory and ensure your CSV files are accessible.

---

## Input Files

The library expects up to five CSV files. Three are required; two are optional enrichments.

---

### 1. `attitudes.csv` (required)

Defines each agent's attitude toward each issue as a signed integer.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `agent` | string | Agent identifier (e.g. `x1`) |
| `<issue_1>` … `<issue_n>` | integer | Attitude value: `-1` (against), `0` (neutral), `+1` (in favour) |

**Example:**

```csv
agent,i1,i2,i3
x1,-1,1,0
x2,1,0,-1
x3,1,-1,-1
```

**Interpretation:**

- Agent `x1` opposes issue `i1`, supports `i2`, and is neutral on `i3`.
- Agent `x2` supports `i1`, is neutral on `i2`, and opposes `i3`.
- Agent `x3` supports `i1`, opposes `i2`, and opposes `i3`.

Any number of agents and issues may be included. Issue column names are arbitrary identifiers and must match those used in `issue_classes.csv` and `issue_names.csv`.

---

### 2. `agent_classes.csv` (required)

Assigns each agent to an ordered importance class. Agents with a lower `order` value are treated as higher priority in the analysis.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `order` | integer | Priority rank (lower = higher priority) |
| `agent` | string | Agent identifier matching `attitudes.csv` |

**Example:**

```csv
order,agent
1,x1
1,x2
2,x3
2,x4
```

**Interpretation:**

- `x1` and `x2` share importance class 1 — the highest priority tier.
- `x3` and `x4` share importance class 2 — a lower priority tier.
- Multiple agents may share the same `order` value; they form one class together.

---

### 3. `issue_classes.csv` (required)

Assigns each issue to an ordered importance class, analogous to `agent_classes.csv`.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `order` | integer | Priority rank (lower = higher priority) |
| `issue` | string | Issue identifier matching attitude column headers |

**Example:**

```csv
order,issue
1,i1
1,i3
2,i2
```

**Interpretation:**

- `i1` and `i3` are in the top importance class.
- `i2` is in a secondary class.

---

### 4. `agent_names.csv` (optional)

Maps short agent identifiers to human-readable display names. When provided, outputs will show friendly names rather than raw identifiers.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `agent` | string | Agent identifier matching `attitudes.csv` |
| `name` | string | Display name for the agent |

**Example:**

```csv
agent,name
x1,Israel
x2,Egypt
```

---

### 5. `issue_names.csv` (optional)

Maps short issue identifiers to human-readable display names.

**Schema:**

| Column | Type | Description |
|---|---|---|
| `issue` | string | Issue identifier matching attitude column headers |
| `name` | string | Display name for the issue |

**Example:**

```csv
issue,name
i1,Autonomous Palestinian state
i2,Jordan River outposts
```

---

## Semantics Reference

The library supports configurable definitions of both **agreement** and **opposition**, allowing the same dataset to be analysed under different theoretical assumptions.

### Agreement Modes

Agreement modes control when agents within a group are considered to agree on an issue.

| Mode | Description |
|---|---|
| `equal` | All agents in the group must have exactly the same attitude value. Strict consensus. |
| `equal_or_zero` | Agents agree if their attitudes are equal, or if some are neutral (`0`). A relaxed form of consensus that tolerates abstention. |

### Opposition Modes

Opposition modes control when two agents' attitudes on an issue are considered opposed.

| Mode | Description |
|---|---|
| `different` | Two attitudes are opposed whenever they differ at all (e.g. `0` vs `1`, `-1` vs `0`). |
| `polarized` | Two attitudes are opposed only when one is strictly positive and the other strictly negative (e.g. `-1` vs `+1`). Neutral positions are never in opposition. |
| `different_nonzero` | Two attitudes are opposed when they differ **and** both are non-zero. Rules out cases where one party is neutral. |

---

## CLI Usage

All functionality is accessed through the command line. The three required CSV files must always be passed. Optional files and analysis flags are added as needed.

### List All Coalitions

Enumerate every valid coalition of agents:

```bash
python pareto_conf_general_csv.py \
    --attitudes attitudes.csv \
    --agent-classes agent_classes.csv \
    --issue-classes issue_classes.csv \
    --coalitions
```

---

### Pareto-Optimal Coalitions under Q\*

Filter coalitions to only those on the Q\* Pareto frontier:

```bash
python pareto_conf_general_csv.py \
    --attitudes attitudes.csv \
    --agent-classes agent_classes.csv \
    --issue-classes issue_classes.csv \
    --coalitions --frontier qstar
```

---

### Enumerate Oppositions for a Group

Analyse opposition relationships relative to a specific group of agents. Pass agent identifiers as a comma-separated list with no spaces:

```bash
python pareto_conf_general_csv.py \
    --attitudes attitudes.csv \
    --agent-classes agent_classes.csv \
    --issue-classes issue_classes.csv \
    --group x1,x2
```

---

### Export Results

Write analysis outputs to files with a shared path prefix. The library appends descriptive suffixes to the prefix you provide:

```bash
python pareto_conf_general_csv.py \
    --attitudes attitudes.csv \
    --agent-classes agent_classes.csv \
    --issue-classes issue_classes.csv \
    --coalitions --group x1 --export-prefix out/results
```

This writes output files such as `out/results_coalitions.csv`, `out/results_oppositions.csv`, etc. into the `out/` directory (which must exist).

---

## Complete CLI Reference

| Flag | Argument | Required | Description |
|---|---|---|---|
| `--attitudes` | `FILE` | Yes | Path to `attitudes.csv` |
| `--agent-classes` | `FILE` | Yes | Path to `agent_classes.csv` |
| `--issue-classes` | `FILE` | Yes | Path to `issue_classes.csv` |
| `--agent-names` | `FILE` | No | Path to `agent_names.csv` for display names |
| `--issue-names` | `FILE` | No | Path to `issue_names.csv` for display names |
| `--coalitions` | — | No | Enumerate all valid coalitions |
| `--frontier` | `qstar` | No | Filter coalitions to the specified Pareto frontier |
| `--group` | `x1,x2,…` | No | Comma-separated agent IDs to analyse for oppositions |
| `--export-prefix` | `PATH` | No | File path prefix for exported result files |
