# Persistent Homology Anchor Graph — Design Document

## Motivation

The fragment-geometry experiments showed that the fixed-threshold kNN graph is
corpus-sensitive. The missing lever looked like **data-driven scale selection**:
letting the fragment space itself determine what similarity threshold defines a
meaningful neighborhood.

Persistent homology looked like the cleanest way to do that without another
hand-tuned threshold.

## Literature Foundation

- **Zhu 2013** — "Persistent homology: an introduction and a new text representation for NLP." Directly applies TDA to text via SIFTS algorithm. Proves persistent homology captures semantic structure in documents.
- **Reininghaus et al. 2015 / Bauer 2021** — "A stable multi-scale kernel for topological machine learning" / "Ripser." Stable persistence diagram kernels + efficient computation. Ripser is ~1000 lines of C++, no dependencies.
- **Battiston et al. 2020** — "Networks beyond pairwise interactions." Reviews simplicial complexes and higher-order networks for multi-way interactions.
- **Xu et al. 2011/2014** — "Efficient Manifold Ranking" (EMR). Anchor graphs scale manifold ranking from O(n²) to O(n), critical for scaling.

## Algorithm: Persistent Homology Scale Selection (PHSS)

### Phase 1: Build the fragment similarity graph

Input: query vector `q`, pool of K documents with fragments, fragment vectors `f_i`.

```
N = total fragments in pool
S[i][j] = cosine(f_i, f_j) for all i, j in [0, N)
```

Cost: O(N² dim) — N is small (pool_size × top_fragments_per_doc ≤ 100 × 6 = 600).

### Phase 2: 0-dimensional persistent homology via Union-Find

0D persistence tracks connected components as the similarity threshold
decreases. It requires only Union-Find.

```python
def compute_0d_persistence(S, N):
    # Sort edges by decreasing similarity
    edges = [(i, j, S[i][j]) for i in range(N) for j in range(i+1, N)]
    edges.sort(key=lambda x: x[2], reverse=True)
    
    uf = UnionFind(N)
    births = [1.0] * N  # birth similarity of each component
    persistence = {}  # component_root -> (birth, death)
    
    for i, j, sim in edges:
        ri = uf.find(i)
        rj = uf.find(j)
        if ri != rj:
            # Younger component dies
            if births[ri] > births[rj]:
                persistence[ri] = (births[ri], sim)
                uf.union(ri, rj)
            else:
                persistence[rj] = (births[rj], sim)
                uf.union(ri, rj)
    
    # Components that survive to the end have infinite persistence
    for root in uf.roots():
        if root not in persistence:
            persistence[root] = (births[root], float('inf'))
    
    return persistence
```

### Phase 3: Scale selection from persistence diagram

The persistence pairs `(birth, death)` form a persistence diagram. For 0D homology:
- `birth` = similarity at which the component first appears (typically 1.0 for single points)
- `death` = similarity at which the component merges with another
- `persistence` = death - birth (or death for single points if birth=1.0)

To select a scale:
```python
def select_scale_from_persistence(persistence):
    # Extract death similarities (merge points)
    deaths = sorted([death for (_, death) in persistence.values() if death != inf])
    
    # Find the largest gap in death similarities
    # This corresponds to the scale where the most components die
    max_gap = 0
    best_idx = 0
    for i in range(len(deaths) - 1):
        gap = deaths[i+1] - deaths[i]
        if gap > max_gap:
            max_gap = gap
            best_idx = i
    
    # The selected scale is between the two death similarities with the largest gap
    selected_scale = (deaths[best_idx] + deaths[best_idx + 1]) / 2
    
    return selected_scale
```

This selects the scale with the clearest separation between robust clusters and
noise.

### Phase 4: Build kNN graph at selected scale and diffuse

```python
def build_graph_at_scale(S, N, scale):
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if S[i][j] >= scale:
                adj[i].append((j, S[i][j]))
                adj[j].append((i, S[i][j]))
    return adj

# Then run the existing diffusion code with this graph
```

## Integration with simeon

### Files to modify
- `benchmarks/bench_vs_reference.cpp`
- `include/simeon/persistent_homology.hpp`
- `src/persistent_homology.cpp`

### Implementation priority
1. Union-Find + 0D persistence (1-2 hours)
2. Scale selection from persistence diagram (30 min)
3. Integration into fragment geometry pipeline (1 hour)
4. Bench on BEIR-3 (2-3 hours)

Total estimated time: 1-2 days.

## Why this looked promising

- fixed `knn=8` and `min_fragment_sim=0.35` were tuned on scifact
- nfcorpus appeared to want coarser neighborhoods
- scifact appeared to want finer ones

Persistent homology promised a data-driven way to choose that scale. At the time
the expected extra cost looked small enough to justify the experiment.

## Open questions

1. Should we use the **largest gap** in death similarities, or the **scale with maximum persistence**, or the **elbow** in the persistence curve?
2. Should we compute persistence on the full fragment pool, or per-document fragment neighborhoods?
3. How does the selected scale vary across queries and corpora? Does it actually adapt as predicted?

These are empirical questions that the first bench run will answer.

---

*2026-04-22. Design document for the next simeon optimization.*
