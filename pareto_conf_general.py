
from __future__ import annotations

"""
Generic qualitative conflict analysis with CSV loading.

This module implements a validated library that can analyze arbitrary conflict 
instances loaded from CSV.

Expected CSV files
==================

1) attitudes.csv  (required)
----------------------------
Matrix of agent attitudes toward issues.

Required columns:
    agent,<issue_1>,<issue_2>,...,<issue_n>

Example:
    agent,i1,i2,i3
    x1,-1,1,0
    x2,1,0,-1
    x3,1,-1,-1

2) agent_classes.csv  (required)
--------------------------------
Ordered importance classes for agents.

Required columns:
    order,agent

Example:
    order,agent
    1,x1
    1,x2
    2,x3
    2,x4

Meaning:
- Lower ``order`` = higher priority.
- All agents with the same order belong to the same importance class.

3) issue_classes.csv  (required)
--------------------------------
Ordered importance classes for issues.

Required columns:
    order,issue

Example:
    order,issue
    1,i1
    1,i3
    2,i2

4) agent_names.csv  (optional)
-------------------------------
Friendly names for agents.

Required columns:
    agent,name

Example:
    agent,name
    x1,Israel
    x2,Egypt

5) issue_names.csv  (optional)
-------------------------------
Friendly names for issues.

Required columns:
    issue,name

Example:
    issue,name
    i1,Autonomous Palestinian state
    i2,Jordan River outposts

Built-in semantics
==================

Agreement modes:
- ``equal``: all attitudes in the group must be exactly equal.
- ``equal_or_zero``: relaxed version where attitudes can be equal or zero (neutral).

Opposition modes:
- ``different``: pairwise attitudes are opposed iff they differ.
- ``polarized``: pairwise attitudes are opposed iff one is positive and the other negative.
- ``different_nonzero``: pairwise attitudes are opposed iff they differ and both are non-zero.

CLI examples
============

List all coalitions:
    python pareto_conf_general_csv.py         --attitudes attitudes.csv         --agent-classes agent_classes.csv         --issue-classes issue_classes.csv         --coalitions

List Pareto-optimal coalitions under Q*:
    python pareto_conf_general_csv.py         --attitudes attitudes.csv         --agent-classes agent_classes.csv         --issue-classes issue_classes.csv         --coalitions --frontier qstar

Enumerate oppositions to group {x1,x2}:
    python pareto_conf_general_csv.py         --attitudes attitudes.csv         --agent-classes agent_classes.csv         --issue-classes issue_classes.csv         --group x1,x2

Export results:
    python pareto_conf_general_csv.py         --attitudes attitudes.csv         --agent-classes agent_classes.csv         --issue-classes issue_classes.csv         --coalitions --group x1 --export-prefix out/results
"""

import argparse
import csv
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Generic, Hashable, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

TAgent = TypeVar("TAgent", bound=Hashable)
TIssue = TypeVar("TIssue", bound=Hashable)


# ============================================================
# GENERIC HELPERS
# ============================================================

def powerset_nonempty(items: Sequence[Hashable]) -> Iterator[FrozenSet[Hashable]]:
    """Yield all non-empty subsets as frozensets."""
    items = list(items)
    for r in range(1, len(items) + 1):
        for comb in combinations(items, r):
            yield frozenset(comb)


def profile(subset: FrozenSet[Hashable], ordered_classes: Sequence[FrozenSet[Hashable]]) -> Tuple[int, ...]:
    """Profile of ``subset`` relative to ordered importance classes."""
    return tuple(len(subset & cls) for cls in ordered_classes)


def mixed_radix_rank(profile_vec: Tuple[int, ...], universe_size: int) -> int:
    """Convert a profile vector to an integer rank preserving lexicographic order."""
    base = universe_size + 1
    rank = 0
    for x in profile_vec:
        rank = rank * base + x
    return rank


def lex_compare(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    """
    Hierarchical lexicographic comparison:
      +1 if a > b
       0 if a == b
      -1 if a < b
    """
    for x, y in zip(a, b):
        if x > y:
            return 1
        if x < y:
            return -1
    return 0


def first_difference_index(a: Tuple[int, ...], b: Tuple[int, ...]) -> Optional[int]:
    """Return the 1-based index of the first coordinate where two vectors differ."""
    for idx, (x, y) in enumerate(zip(a, b), start=1):
        if x != y:
            return idx
    return None


def pareto_frontier(items: Iterable[object], dominates: Callable[[object, object], bool]) -> List[object]:
    """
    Generic O(n^2) Pareto frontier:
    keep x iff there is no y such that y dominates x.

    NOTE: This is retained for reference only. All internal callers have been
    replaced with O(n log n) specialised implementations.
    """
    items = list(items)
    frontier = []
    for i, x in enumerate(items):
        if not any(i != j and dominates(y, x) for j, y in enumerate(items)):
            frontier.append(x)
    return frontier


def skyline_2d_max(items: Iterable[object], key_x: Callable[[object], int], key_y: Callable[[object], int]) -> List[object]:
    """
    2D skyline for maximizing (key_x, key_y).

    Correct tie handling:
    - duplicates with identical (x, y) are preserved,
    - within the same x-group, only top-y ties survive,
    - a group is discarded if a strictly larger x-group already reaches the same or higher y.
    """
    ordered = sorted(items, key=lambda o: (-key_x(o), -key_y(o)))
    if not ordered:
        return []

    frontier: List[object] = []
    max_y_from_strictly_larger_x: Optional[int] = None

    i = 0
    while i < len(ordered):
        current_x = key_x(ordered[i])
        group = []
        while i < len(ordered) and key_x(ordered[i]) == current_x:
            group.append(ordered[i])
            i += 1

        group_max_y = key_y(group[0])
        if max_y_from_strictly_larger_x is None or group_max_y > max_y_from_strictly_larger_x:
            frontier.extend(item for item in group if key_y(item) == group_max_y)

        if max_y_from_strictly_larger_x is None or group_max_y > max_y_from_strictly_larger_x:
            max_y_from_strictly_larger_x = group_max_y

    return frontier


# ============================================================
# DOMAIN OBJECTS
# ============================================================

@dataclass(frozen=True)
class Structure(Generic[TAgent, TIssue]):
    """
    Generic pair (A, B):
      - coalition-like structure: A = subset of agents, B = subset of issues
      - opposition-like structure: A = opposing agents, B = disagreement issues
    """
    A: FrozenSet[TAgent]
    B: FrozenSet[TIssue]
    vU: Tuple[int, ...]
    vI: Tuple[int, ...]
    rU: int
    rI: int


@dataclass(frozen=True)
class BiConflict(Generic[TAgent, TIssue]):
    coalition: Structure[TAgent, TIssue]
    opposition: Structure[TAgent, TIssue]


# ============================================================
# BUILT-IN SEMANTICS
# ============================================================

AgreementRule = Callable[[Sequence[int]], bool]
OppositionRule = Callable[[int, int, bool], bool]


def agreement_equal(values: Sequence[int]) -> bool:
    return len(set(values)) == 1

def agreement_equal_or_zero(values: Sequence[int]) -> bool:
    vs = set(values)-{0}
    return len(set(vs)) == 1


def opposition_different(v1: int, v2: int) -> bool:
    return v1 != v2


def opposition_polarized(v1: int, v2: int) -> bool:
    return v1 * v2 < 0 or ((v1 == 0) ^ (v2 == 0))


def opposition_different_nonzero(v1: int, v2: int) -> bool:
    if v1 == 0 or v2 == 0:
        return False
    return v1 != v2


AGREEMENT_MODES: Dict[str, AgreementRule] = {
    "equal": agreement_equal,
    "equal_or_zero": agreement_equal_or_zero,
}

OPPOSITION_MODES: Dict[str, OppositionRule] = {
    "different": opposition_different,
    "polarized": opposition_polarized,
    "different_nonzero": opposition_different_nonzero,
}


# ============================================================
# GENERIC CONFLICT MODEL
# ============================================================

@dataclass
class ConflictModel(Generic[TAgent, TIssue]):
    agents: Sequence[TAgent]
    issues: Sequence[TIssue]
    attitudes: Dict[TAgent, Dict[TIssue, int]]
    agent_classes: Sequence[FrozenSet[TAgent]]  # ordered, highest priority first
    issue_classes: Sequence[FrozenSet[TIssue]]  # ordered, highest priority first
    agent_names: Dict[TAgent, str] = field(default_factory=dict)
    issue_names: Dict[TIssue, str] = field(default_factory=dict)
    agreement_rule: AgreementRule = agreement_equal
    opposition_rule: OppositionRule = opposition_different

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_csv(
        cls,
        attitudes_csv: str | Path,
        agent_classes_csv: str | Path,
        issue_classes_csv: str | Path,
        *,
        agent_names_csv: Optional[str | Path] = None,
        issue_names_csv: Optional[str | Path] = None,
        agreement_mode: str = "equal",
        opposition_mode: str = "different",
        delimiter: str = ",",
        encoding: str = "utf-8-sig",
    ) -> "ConflictModel[str, str]":
        """Load a conflict model from CSV files."""
        attitudes, agents, issues = _read_attitudes_csv(attitudes_csv, delimiter=delimiter, encoding=encoding)
        agent_classes = _read_agent_classes_csv(agent_classes_csv, known_agents=set(agents), delimiter=delimiter, encoding=encoding)
        issue_classes = _read_issue_classes_csv(issue_classes_csv, known_issues=set(issues), delimiter=delimiter, encoding=encoding)
        agent_names = _read_name_map_csv(agent_names_csv, key_col="agent", delimiter=delimiter, encoding=encoding) if agent_names_csv else {}
        issue_names = _read_name_map_csv(issue_names_csv, key_col="issue", delimiter=delimiter, encoding=encoding) if issue_names_csv else {}

        if agreement_mode not in AGREEMENT_MODES:
            raise ValueError(f"Unknown agreement_mode={agreement_mode!r}. Choose one of {sorted(AGREEMENT_MODES)}")
        if opposition_mode not in OPPOSITION_MODES:
            raise ValueError(f"Unknown opposition_mode={opposition_mode!r}. Choose one of {sorted(OPPOSITION_MODES)}")

        return cls(
            agents=agents,
            issues=issues,
            attitudes=attitudes,
            agent_classes=agent_classes,
            issue_classes=issue_classes,
            agent_names=agent_names,
            issue_names=issue_names,
            agreement_rule=AGREEMENT_MODES[agreement_mode],
            opposition_rule=OPPOSITION_MODES[opposition_mode],
        )

    def _validate(self) -> None:
        agent_set = set(self.agents)
        issue_set = set(self.issues)

        if not self.agents:
            raise ValueError("agents must not be empty")
        if not self.issues:
            raise ValueError("issues must not be empty")

        if set(self.attitudes.keys()) != agent_set:
            missing = agent_set - set(self.attitudes.keys())
            extra = set(self.attitudes.keys()) - agent_set
            raise ValueError(f"attitudes keys must match agents exactly; missing={missing}, extra={extra}")

        for a in self.agents:
            row_issues = set(self.attitudes[a].keys())
            if row_issues != issue_set:
                missing = issue_set - row_issues
                extra = row_issues - issue_set
                raise ValueError(f"attitudes[{a}] must cover all issues exactly; missing={missing}, extra={extra}")

        for idx, cls in enumerate(self.agent_classes, start=1):
            unknown = set(cls) - agent_set
            if unknown:
                raise ValueError(f"agent_classes[{idx}] contains unknown agents: {unknown}")

        for idx, cls in enumerate(self.issue_classes, start=1):
            unknown = set(cls) - issue_set
            if unknown:
                raise ValueError(f"issue_classes[{idx}] contains unknown issues: {unknown}")

        flat_agents = [a for cls in self.agent_classes for a in cls]
        flat_issues = [i for cls in self.issue_classes for i in cls]

        if set(flat_agents) != agent_set:
            missing = agent_set - set(flat_agents)
            extra = set(flat_agents) - agent_set
            raise ValueError(f"Agent classes must cover each agent exactly once; missing={missing}, extra={extra}")
        if len(flat_agents) != len(set(flat_agents)):
            raise ValueError("Agent classes must be disjoint; at least one agent appears in multiple classes")

        if set(flat_issues) != issue_set:
            missing = issue_set - set(flat_issues)
            extra = set(flat_issues) - issue_set
            raise ValueError(f"Issue classes must cover each issue exactly once; missing={missing}, extra={extra}")
        if len(flat_issues) != len(set(flat_issues)):
            raise ValueError("Issue classes must be disjoint; at least one issue appears in multiple classes")

    def make_structure(self, A: FrozenSet[TAgent], B: FrozenSet[TIssue]) -> Structure[TAgent, TIssue]:
        vU = profile(A, self.agent_classes)
        vI = profile(B, self.issue_classes)
        rU = mixed_radix_rank(vU, len(self.agents))
        rI = mixed_radix_rank(vI, len(self.issues))
        return Structure(A=A, B=B, vU=vU, vI=vI, rU=rU, rI=rI)

    def common_agreement_issues(self, group: FrozenSet[TAgent]) -> FrozenSet[TIssue]:
        """Return the maximal set of issues on which all agents in ``group`` agree."""
        if not group:
            raise ValueError("group must be non-empty")

        agreed = set()
        for issue in self.issues:
            values = [self.attitudes[a][issue] for a in group]
            if self.agreement_rule(values):
                agreed.add(issue)
        return frozenset(agreed)

    def enumerate_all_coalitions(self) -> List[Structure[TAgent, TIssue]]:
        """
        Enumerate all valid coalition structures (X, B):
          - X is any non-empty subset of agents
          - B is any non-empty subset of the maximal agreement set of X
        """
        out: List[Structure[TAgent, TIssue]] = []
        for X in powerset_nonempty(self.agents):
            Xf = frozenset(X)
            Bmax = self.common_agreement_issues(Xf)
            if not Bmax:
                continue
            for B in powerset_nonempty(sorted(Bmax, key=str)):
                out.append(self.make_structure(Xf, frozenset(B)))
        return out

    def rho(self, x: TAgent, y: TAgent) -> FrozenSet[TIssue]:
        """
        Pairwise opposition set between x and y:
        all issues on which their attitudes satisfy the opposition rule.
        """
        out = {
            issue
            for issue in self.issues
            if self.opposition_rule(self.attitudes[x][issue], self.attitudes[y][issue])
        }
        return frozenset(out)

    def group_opposition_issues(self, X: FrozenSet[TAgent], G: FrozenSet[TAgent]) -> FrozenSet[TIssue]:
        """B_G(X) = intersection over x in X and g in G of rho(x, g)."""
        if not X:
            raise ValueError("X must be non-empty")
        if not G:
            raise ValueError("G must be non-empty")

        current = set(self.issues)
        for x in X:
            for g in G:
                current &= set(self.rho(x, g))
                if not current:
                    return frozenset()
        return frozenset(current)

    def enumerate_oppositions_for_group(self, G: FrozenSet[TAgent]) -> List[Structure[TAgent, TIssue]]:
        """
        Enumerate all oppositions to a reference group G:
          - X is any non-empty subset of U minus G
          - C = B_G(X)
          - keep only non-empty C
        """
        if not G:
            raise ValueError("G must be non-empty")
        if not set(G).issubset(set(self.agents)):
            unknown = set(G) - set(self.agents)
            raise ValueError(f"G contains unknown agents: {unknown}")

        remaining = [a for a in self.agents if a not in G]
        out: List[Structure[TAgent, TIssue]] = []
        for X in powerset_nonempty(remaining):
            Xf = frozenset(X)
            C = self.group_opposition_issues(Xf, G)
            if C:
                out.append(self.make_structure(Xf, C))
        return out

    def enumerate_biconflicts_for_group(self, G: FrozenSet[TAgent]) -> List[BiConflict[TAgent, TIssue]]:
        """
        Build bi-conflicts for a fixed reference coalition G by pairing the coalition
        structure (G, B) with each admissible opposition structure (X, C), where B is
        any non-empty subset of the maximal agreement set of G.
        """
        Bmax = self.common_agreement_issues(G)
        if not Bmax:
            return []

        coalitions = [self.make_structure(G, frozenset(B)) for B in powerset_nonempty(sorted(Bmax, key=str))]
        # coalitions = [self.make_structure(G, frozenset(Bmax))]
        oppositions = self.enumerate_oppositions_for_group(G)
        return [BiConflict(coalition=c, opposition=o) for c in coalitions for o in oppositions]

    def fmt_agents(self, A: FrozenSet[TAgent]) -> str:
        return "{" + ", ".join(self.agent_names.get(a, str(a)) for a in sorted(A, key=str)) + "}"

    def fmt_issues(self, B: FrozenSet[TIssue]) -> str:
        return "{" + ", ".join(self.issue_names.get(i, str(i)) for i in sorted(B, key=str)) + "}"

    def describe_structure(self, s: Structure[TAgent, TIssue]) -> str:
        return f"A={self.fmt_agents(s.A)}, B={self.fmt_issues(s.B)}, vU={s.vU}, vI={s.vI}, rU={s.rU}, rI={s.rI}"

    def describe_biconflict(self, bc: BiConflict[TAgent, TIssue]) -> str:
        return f"coalition=[{self.describe_structure(bc.coalition)}]; opposition=[{self.describe_structure(bc.opposition)}]"


# ============================================================
# QUALITATIVE DOMINANCE
# ============================================================

def dominates_Q(a: Structure, b: Structure) -> bool:
    """
    Definition 9:
    (A,B) >>_Q (C,D)
    """
    cu = lex_compare(a.vU, b.vU)
    ci = lex_compare(a.vI, b.vI)
    return cu >= 0 and ci >= 0 and (cu > 0 or ci > 0)


def dominates_Q_star(a: Structure, b: Structure) -> bool:
    """
    Definition 10:
    (A,B) >>*_Q (C,D)
    """
    if dominates_Q(a, b):
        return True

    cu = lex_compare(a.vU, b.vU)
    ci = lex_compare(a.vI, b.vI)

    i = first_difference_index(a.vU, b.vU)
    j = first_difference_index(a.vI, b.vI)

    if i is None or j is None:
        return False

    # a better on agents, b better on issues, but agent advantage appears earlier
    if cu > 0 and ci < 0 and i < j:
        return True

    # b better on agents, a better on issues, but issue advantage appears earlier
    if cu < 0 and ci > 0 and j < i:
        return True

    return False


def frontier_Q(structures: Iterable[Structure]) -> List[Structure]:
    """
    Fast frontier under >>_Q using rank-based 2D skyline.
    Valid because ranks preserve the lexicographic ordering of profiles.
    """
    return skyline_2d_max(structures, key_x=lambda s: s.rU, key_y=lambda s: s.rI)  # type: ignore[arg-type]


def frontier_Q_star(structures: Iterable[Structure]) -> List[Structure]:
    """
    O(n log n) frontier under >>*_Q.

    >>*_Q extends >>_Q with two asymmetric trade-off cases that can dominate
    even when one rank is strictly worse:
      (T1) rU(a) > rU(b)  and  rI(a) < rI(b)  and  first_diff_U < first_diff_I
      (T2) rU(a) < rU(b)  and  rI(a) > rI(b)  and  first_diff_I < first_diff_U

    Strategy:
      1. Run the O(n log n) 2D skyline to get the >>_Q frontier F.
         Any item not in F is dominated by something that is also >= on *both*
         dimensions, which is already stronger than any >>*_Q domination, so
         non-frontier items under >>_Q cannot survive under >>*_Q either.
      2. Within the small set F, apply the full >>*_Q pairwise check.
         In practice |F| << n, so this step is negligible.
    """
    items = list(structures)
    # Step 1: cheap pre-filter via the weaker >>_Q skyline.
    candidates: List[Structure] = skyline_2d_max(items, key_x=lambda s: s.rU, key_y=lambda s: s.rI)  # type: ignore[arg-type]
    # Step 2: pairwise check only among the small frontier.
    n = len(candidates)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i != j and not dominated[j] and dominates_Q_star(candidates[j], candidates[i]):
                dominated[i] = True
                break
    return [s for s, d in zip(candidates, dominated) if not d]


# ============================================================
# BI-CONFLICT DOMINANCE
# ============================================================

def dominates_bi_Q(a: BiConflict, b: BiConflict) -> bool:
    coalition_dominance = dominates_Q(a.coalition, b.coalition)
    opposition_dominance = dominates_Q(a.opposition, b.opposition)
    if a.coalition == b.coalition and opposition_dominance:
        return True  # opposition dominate 
    if a.opposition == b.opposition and coalition_dominance:
        return True  # coalition dominate
    return coalition_dominance and opposition_dominance  # both must dominate
    # return dominates_Q(a.coalition, b.coalition) and dominates_Q(a.opposition, b.opposition)


def dominates_bi_Q_star(a: BiConflict, b: BiConflict) -> bool:
    coalition_dominance = dominates_Q_star(a.coalition, b.coalition)
    opposition_dominance = dominates_Q_star(a.opposition, b.opposition)
    if a.coalition == b.coalition and opposition_dominance:
        return True  # opposition dominate 
    if a.opposition == b.opposition and coalition_dominance:
        return True  # coalition dominate
    return coalition_dominance and opposition_dominance  # both must dominate


def _struct_frontier_for(
    structures: Iterable[Structure],
    struct_dominates: Callable[[Structure, Structure], bool],
) -> List[Structure]:
    """Return the Pareto frontier of *structures* under *struct_dominates*."""
    if struct_dominates is dominates_Q:
        return skyline_2d_max(list(structures), key_x=lambda s: s.rU, key_y=lambda s: s.rI)  # type: ignore[arg-type]
    return frontier_Q_star(list(structures))


def _biconflict_frontier(
    biconflicts: Iterable[BiConflict],
    struct_dominates: Callable[[Structure, Structure], bool],
) -> List[BiConflict]:
    """
    O(n log n) biconflict frontier for dominance relations of the form:
        a dom b  iff  (a.coal dom b.coal AND a.opp dom b.opp)
                   OR (a.coal == b.coal AND a.opp dom b.opp)
                   OR (a.opp  == b.opp  AND a.coal dom b.coal)

    Algorithm
    ---------
    1. Compute the Pareto frontier of *all* coalition components — O(n log n).
       A biconflict (c, o) can only be dominated by some (c', o') where c' dom c
       or c' == c.  If c is not on the coalition frontier there exists a c'' with
       c'' strictly dom c; any (c'', o'') with o'' dom o (or o'' == o) then
       dominates (c, o) regardless of o.  However, the "same component" special
       cases mean we cannot independently discard by one dimension only, so we use
       the frontier membership as a *pre-filter*: keep a biconflict if its
       coalition OR its opposition survives its respective frontier.
    2. Within the much smaller candidate set, apply a pairwise dominance check.
    """
    items = list(biconflicts)
    if not items:
        return []

    # Build id-based membership sets for fast lookup.
    coal_front_ids: set = {id(s) for s in _struct_frontier_for(
        (bc.coalition for bc in items), struct_dominates
    )}
    opp_front_ids: set = {id(s) for s in _struct_frontier_for(
        (bc.opposition for bc in items), struct_dominates
    )}

    # Pre-filter: keep biconflicts where at least one component is on its frontier.
    candidates = [
        bc for bc in items
        if id(bc.coalition) in coal_front_ids or id(bc.opposition) in opp_front_ids
    ]

    # Pairwise check among the small candidate set.
    n = len(candidates)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            a, b = candidates[j], candidates[i]
            cd = struct_dominates(a.coalition, b.coalition)
            od = struct_dominates(a.opposition, b.opposition)
            same_coal = a.coalition == b.coalition
            same_opp  = a.opposition == b.opposition
            if (same_coal and od) or (same_opp and cd) or (cd and od):
                dominated[i] = True
                break

    return [bc for bc, d in zip(candidates, dominated) if not d]


def biconflict_frontier_Q(biconflicts: Iterable[BiConflict]) -> List[BiConflict]:
    return _biconflict_frontier(biconflicts, dominates_Q)  # type: ignore[arg-type]


def biconflict_frontier_Q_star(biconflicts: Iterable[BiConflict]) -> List[BiConflict]:
    return _biconflict_frontier(biconflicts, dominates_Q_star)  # type: ignore[arg-type]




# ============================================================
# CSV I/O HELPERS
# ============================================================

def _read_csv_dicts(path: str | Path, *, delimiter: str = ",", encoding: str = "utf-8-sig") -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    with p.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {p}")
        return [dict(row) for row in reader]


def _read_attitudes_csv(path: str | Path, *, delimiter: str = ",", encoding: str = "utf-8-sig") -> Tuple[Dict[str, Dict[str, int]], List[str], List[str]]:
    p = Path(path)
    with p.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Attitudes CSV has no header: {p}")
        if "agent" not in reader.fieldnames:
            raise ValueError(f"Attitudes CSV must contain an 'agent' column: {p}")

        issues = [c for c in reader.fieldnames if c != "agent"]
        if not issues:
            raise ValueError(f"Attitudes CSV must define at least one issue column after 'agent': {p}")

        attitudes: Dict[str, Dict[str, int]] = {}
        agents: List[str] = []
        for row_num, row in enumerate(reader, start=2):
            agent = (row.get("agent") or "").strip()
            if not agent:
                raise ValueError(f"Missing agent id at {p}:{row_num}")
            if agent in attitudes:
                raise ValueError(f"Duplicate agent {agent!r} in {p}:{row_num}")

            attitudes[agent] = {}
            agents.append(agent)
            for issue in issues:
                raw = row.get(issue, "")
                if raw is None or str(raw).strip() == "":
                    raise ValueError(f"Missing value for agent={agent!r}, issue={issue!r} at {p}:{row_num}")
                try:
                    attitudes[agent][issue] = int(str(raw).strip())
                except ValueError as e:
                    raise ValueError(f"Attitude values must be integers; got {raw!r} for agent={agent!r}, issue={issue!r} at {p}:{row_num}") from e

    return attitudes, agents, issues


def _group_ordered_classes(rows: List[Dict[str, str]], *, order_col: str, item_col: str, known_items: set[str], label: str) -> List[FrozenSet[str]]:
    grouped: Dict[int, List[str]] = {}
    seen: set[str] = set()

    for row_num, row in enumerate(rows, start=2):
        if order_col not in row or item_col not in row:
            raise ValueError(f"{label} CSV must contain columns {order_col!r} and {item_col!r}")

        item = (row.get(item_col) or "").strip()
        if not item:
            raise ValueError(f"Empty {item_col!r} value in {label} CSV at data row {row_num}")
        if item not in known_items:
            raise ValueError(f"Unknown {item_col} {item!r} in {label} CSV at data row {row_num}")
        if item in seen:
            raise ValueError(f"Duplicate {item_col} {item!r} in {label} CSV at data row {row_num}")
        seen.add(item)

        try:
            order = int((row.get(order_col) or "").strip())
        except ValueError as e:
            raise ValueError(f"Invalid integer order for {item_col}={item!r} in {label} CSV at data row {row_num}") from e

        grouped.setdefault(order, []).append(item)

    if seen != known_items:
        missing = known_items - seen
        extra = seen - known_items
        raise ValueError(f"{label} CSV must cover all items exactly once; missing={missing}, extra={extra}")

    return [frozenset(grouped[k]) for k in sorted(grouped)]


def _read_agent_classes_csv(path: str | Path, *, known_agents: set[str], delimiter: str = ",", encoding: str = "utf-8-sig") -> List[FrozenSet[str]]:
    rows = _read_csv_dicts(path, delimiter=delimiter, encoding=encoding)
    return _group_ordered_classes(rows, order_col="order", item_col="agent", known_items=known_agents, label="agent_classes")


def _read_issue_classes_csv(path: str | Path, *, known_issues: set[str], delimiter: str = ",", encoding: str = "utf-8-sig") -> List[FrozenSet[str]]:
    rows = _read_csv_dicts(path, delimiter=delimiter, encoding=encoding)
    return _group_ordered_classes(rows, order_col="order", item_col="issue", known_items=known_issues, label="issue_classes")


def _read_name_map_csv(path: str | Path, *, key_col: str, delimiter: str = ",", encoding: str = "utf-8-sig") -> Dict[str, str]:
    rows = _read_csv_dicts(path, delimiter=delimiter, encoding=encoding)
    out: Dict[str, str] = {}
    for row_num, row in enumerate(rows, start=2):
        if key_col not in row or "name" not in row:
            raise ValueError(f"Name CSV must contain columns {key_col!r} and 'name'")
        key = (row.get(key_col) or "").strip()
        name = (row.get("name") or "").strip()
        if not key:
            raise ValueError(f"Empty {key_col!r} value in name CSV at data row {row_num}")
        if key in out:
            raise ValueError(f"Duplicate {key_col} {key!r} in name CSV at data row {row_num}")
        out[key] = name or key
    return out


def export_structures_csv(path: str | Path, structures: Sequence[Structure], model: ConflictModel, *, delimiter: str = ",", encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["A", "B", "vU", "vI", "rU", "rI", "A_names", "B_names"])
        for s in structures:
            writer.writerow([
                "|".join(map(str, sorted(s.A, key=str))),
                "|".join(map(str, sorted(s.B, key=str))),
                "|".join(map(str, s.vU)),
                "|".join(map(str, s.vI)),
                s.rU,
                s.rI,
                "|".join(model.agent_names.get(a, str(a)) for a in sorted(s.A, key=str)),
                "|".join(model.issue_names.get(i, str(i)) for i in sorted(s.B, key=str)),
            ])


def export_biconflicts_csv(path: str | Path, biconflicts: Sequence[BiConflict], model: ConflictModel, *, delimiter: str = ",", encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow([
            "coal_A", "coal_B", "coal_vU", "coal_vI", "coal_rU", "coal_rI",
            "opp_A", "opp_B", "opp_vU", "opp_vI", "opp_rU", "opp_rI",
        ])
        for bc in biconflicts:
            c = bc.coalition
            o = bc.opposition
            writer.writerow([
                "|".join(map(str, sorted(c.A, key=str))),
                "|".join(map(str, sorted(c.B, key=str))),
                "|".join(map(str, c.vU)),
                "|".join(map(str, c.vI)),
                c.rU,
                c.rI,
                "|".join(map(str, sorted(o.A, key=str))),
                "|".join(map(str, sorted(o.B, key=str))),
                "|".join(map(str, o.vU)),
                "|".join(map(str, o.vI)),
                o.rU,
                o.rI,
            ])


# ============================================================
# CLI
# ============================================================

def _print_structures(title: str, structures: Sequence[Structure], model: ConflictModel) -> None:
    print(title)
    print("-" * len(title))
    print(f"count = {len(structures)}")
    for idx, s in enumerate(structures, start=1):
        print(f"[{idx}] {model.describe_structure(s)}")
    print()


def _print_biconflicts(title: str, biconflicts: Sequence[BiConflict], model: ConflictModel) -> None:
    print(title)
    print("-" * len(title))
    print(f"count = {len(biconflicts)}")
    for idx, bc in enumerate(biconflicts, start=1):
        print(f"[{idx}] {model.describe_biconflict(bc)}")
    print()


def _parse_group(text: str) -> FrozenSet[str]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items:
        raise ValueError("--group must contain at least one agent id, e.g. x1,x2")
    return frozenset(items)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic qualitative conflict analysis loaded from CSV.")
    parser.add_argument("--attitudes", required=True, help="Path to attitudes.csv")
    parser.add_argument("--agent-classes", required=True, help="Path to agent_classes.csv")
    parser.add_argument("--issue-classes", required=True, help="Path to issue_classes.csv")
    parser.add_argument("--agent-names", help="Optional path to agent_names.csv")
    parser.add_argument("--issue-names", help="Optional path to issue_names.csv")
    parser.add_argument("--agreement-mode", default="equal", choices=sorted(AGREEMENT_MODES), help="Agreement semantics")
    parser.add_argument("--opposition-mode", default="different", choices=sorted(OPPOSITION_MODES), help="Opposition semantics")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',')")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig)")
    parser.add_argument("--coalitions", action="store_true", help="Enumerate all admissible coalitions")
    parser.add_argument("--group", help="Reference group for opposition / bi-conflict analysis (comma-separated ids, e.g. x1,x2)")
    parser.add_argument("--frontier", choices=["q", "qstar", "both", "none"], default="none", help="Which frontier to compute for the requested objects")
    parser.add_argument("--biconflicts", action="store_true", help="When used with --group, enumerate bi-conflicts for that group")
    parser.add_argument("--export-prefix", help="Optional prefix for exporting CSV results, e.g. out/results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model = ConflictModel.from_csv(
        attitudes_csv=args.attitudes,
        agent_classes_csv=args.agent_classes,
        issue_classes_csv=args.issue_classes,
        agent_names_csv=args.agent_names,
        issue_names_csv=args.issue_names,
        agreement_mode=args.agreement_mode,
        opposition_mode=args.opposition_mode,
        delimiter=args.delimiter,
        encoding=args.encoding,
    )

    print("Loaded conflict model")
    print("---------------------")
    print(f"agents = {len(model.agents)} :: {', '.join(map(str, model.agents))}")
    print(f"issues = {len(model.issues)} :: {', '.join(map(str, model.issues))}")
    print(f"agent classes = {model.agent_classes}")
    print(f"issue classes = {model.issue_classes}")
    print()

    if not args.coalitions and not args.group:
        parser.error("Nothing to do: specify at least one of --coalitions or --group")

    if args.coalitions:
        coalitions = model.enumerate_all_coalitions()
        _print_structures("All coalitions", coalitions, model)

        q_front = None
        qstar_front = None
        if args.frontier in {"q", "both"}:
            q_front = frontier_Q(coalitions)
            _print_structures("Coalition frontier under Q", q_front, model)
        if args.frontier in {"qstar", "both"}:
            qstar_front = frontier_Q_star(coalitions)
            _print_structures("Coalition frontier under Q*", qstar_front, model)

        if args.export_prefix:
            prefix = Path(args.export_prefix)
            export_structures_csv(prefix.with_name(prefix.name + "_coalitions.csv"), coalitions, model, delimiter=args.delimiter)
            if q_front is not None:
                export_structures_csv(prefix.with_name(prefix.name + "_coalitions_frontier_q.csv"), q_front, model, delimiter=args.delimiter)
            if qstar_front is not None:
                export_structures_csv(prefix.with_name(prefix.name + "_coalitions_frontier_qstar.csv"), qstar_front, model, delimiter=args.delimiter)

    if args.group:
        G = _parse_group(args.group)
        oppositions = model.enumerate_oppositions_for_group(G)
        _print_structures(f"Oppositions to group {model.fmt_agents(G)}", oppositions, model)

        q_front = None
        qstar_front = None
        if args.frontier in {"q", "both"}:
            q_front = frontier_Q(oppositions)
            _print_structures(f"Opposition frontier under Q for {model.fmt_agents(G)}", q_front, model)
        if args.frontier in {"qstar", "both"}:
            qstar_front = frontier_Q_star(oppositions)
            _print_structures(f"Opposition frontier under Q* for {model.fmt_agents(G)}", qstar_front, model)

        if args.export_prefix:
            prefix = Path(args.export_prefix)
            export_structures_csv(prefix.with_name(prefix.name + "_oppositions.csv"), oppositions, model, delimiter=args.delimiter)
            if q_front is not None:
                export_structures_csv(prefix.with_name(prefix.name + "_oppositions_frontier_q.csv"), q_front, model, delimiter=args.delimiter)
            if qstar_front is not None:
                export_structures_csv(prefix.with_name(prefix.name + "_oppositions_frontier_qstar.csv"), qstar_front, model, delimiter=args.delimiter)

        if args.biconflicts:
            biconflicts = model.enumerate_biconflicts_for_group(G)
            _print_biconflicts(f"Bi-conflicts for group {model.fmt_agents(G)}", biconflicts, model)

            bi_front_q = None
            bi_front_qstar = None
            if args.frontier in {"q", "both"}:
                bi_front_q = biconflict_frontier_Q(biconflicts)
                _print_biconflicts(f"Bi-conflict frontier under Q for {model.fmt_agents(G)}", bi_front_q, model)
            if args.frontier in {"qstar", "both"}:
                bi_front_qstar = biconflict_frontier_Q_star(biconflicts)
                _print_biconflicts(f"Bi-conflict frontier under Q* for {model.fmt_agents(G)}", bi_front_qstar, model)

            if args.export_prefix:
                prefix = Path(args.export_prefix)
                export_biconflicts_csv(prefix.with_name(prefix.name + "_biconflicts.csv"), biconflicts, model, delimiter=args.delimiter)
                if bi_front_q is not None:
                    export_biconflicts_csv(prefix.with_name(prefix.name + "_biconflicts_frontier_q.csv"), bi_front_q, model, delimiter=args.delimiter)
                if bi_front_qstar is not None:
                    export_biconflicts_csv(prefix.with_name(prefix.name + "_biconflicts_frontier_qstar.csv"), bi_front_qstar, model, delimiter=args.delimiter)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
