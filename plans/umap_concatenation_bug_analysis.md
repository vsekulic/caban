# UMAP Concatenation Bug Analysis

## Error Message
```
Error in UMAPAnalysis for G06: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 413 and the array at index 1 has size 434
```

## Root Cause

The bug is in the [`UMAPAnalysis.fit_umap_concatenated()`](SSTCa2_population.py:1452) method at line 1502:

```python
S_concat = np.hstack(S_list)
```

### The Problem

`np.hstack()` concatenates arrays **horizontally** (along axis 1 / columns). For this to work, **all arrays must have the same number of rows** (axis 0 dimension = number of cells/neurons).

The code iterates over sessions and collects their `S` matrices:

```python
for name, sess in self.sessions.items():
    S = sess.S  # <-- Different sessions have different number of cells!
    # ... normalization ...
    S_list.append(S_norm)

S_concat = np.hstack(S_list)  # <-- Fails because S_list arrays have different row counts
```

Each session (`TFC_cond`, `Test_B`, `Test_B_1wk`) has a `.S` attribute with shape `(n_cells, n_frames)`, where `n_cells` is the **raw number of detected cells** in that session. These numbers differ across sessions:
- `TFC_cond.S` might have 413 cells
- `Test_B.S` might have 434 cells

When `np.hstack()` tries to concatenate these along axis 1, it requires all arrays to have the same number of rows (413 == 434), which fails.

### Why Cross-Registration Doesn't Help Here

The session objects do have a `crossreg` attribute (`CrossRegMapping`) that maps cells across sessions, and the `get_S_mapping()` method can extract the **common cell subset**. However, `fit_umap_concatenated()` accesses `sess.S` directly without applying cross-registration filtering.

## Affected Code Paths

1. **[`run_population_pca_pipeline()`](SSTCa2_population.py:1544)** calls `fit_umap_concatenated()` at line 1864:
   ```python
   embedding_concat, session_boundaries = umap_analysis.fit_umap_concatenated(...)
   ```

2. This is triggered when `len(sessions_dict) >= 2` (line 1863), meaning when both Test_B and Test_B_1wk sessions are available.

## Fix Required

The `fit_umap_concatenated()` method needs to:
1. Use cross-registered cell subsets instead of raw `.S` arrays
2. Extract only the cells common across all sessions before concatenating

This requires either:
- **Option A**: Modify the method to accept a `crossreg` parameter and use `sess.get_S_mapping()` to get the common cell subset
- **Option B**: Store cross-registered `.S` on session objects (like how other analyses work)
- **Option C**: Find the intersection of cell indices across all sessions and use that subset

## Similar Bug in `fit_umap_crossreg()`

The [`fit_umap_crossreg()`](SSTCa2_population.py:1402) method at line 1447 does **not** have this bug because it fits **separate UMAP models per session** (no concatenation across sessions):

```python
for name, sess in self.sessions.items():
    S = sess.S
    # ... normalize ...
    embeddings[name] = umap_obj.fit_transform(S_norm.T)  # Separate per session
```

## Proposed Fix (Option A - Crossreg Parameter)

In `run_population_pca_pipeline()`, pass the crossreg mapping to `UMAPAnalysis`:

```python
umap_analysis = UMAPAnalysis(umap_sessions, mouse_id, group, config)
umap_analysis.crossreg = TFC_B_B_1wk_crossreg  # Add crossreg reference

# Then in fit_umap_concatenated:
for name, sess in self.sessions.items():
    if not hasattr(sess, 'S'):
        continue
    
    # Get cross-registered cell subset
    mapping_type = self.crossreg.get_mappings_for_groups('TFC_cond+Test_B+Test_B_1wk')
    S_common, _, _, _ = sess.get_S_mapping(mapping_type)  # Common cells only
    
    # Normalize and add to list
    # ...
```

## Summary

| Aspect | Details |
|--------|---------|
| **Bug Location** | [`SSTCa2_population.py:1502`](SSTCa2_population.py:1502) |
| **Error Type** | `ValueError` - shape mismatch in `np.hstack()` |
| **Root Cause** | Sessions have different raw cell counts; no crossreg filtering before concatenation |
| **Affected Mice** | All mice with multiple sessions (G06 has 413 vs 434 cells) |
| **Fix Complexity** | Low - need to add crossreg filtering before `np.hstack()` |
