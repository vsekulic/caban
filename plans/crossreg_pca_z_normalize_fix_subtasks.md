# CrossregPCA z_normalize() Fix - Subtasks

## Overview

This document breaks down the plan in `plans/crossreg_pca_z_normalize_fix.md` into discrete, actionable subtasks. Each subtask is designed to be completed independently and tested before moving to the next.

---

## Subtask 1: Add `z_normalize()` Method to `CrossregPCA`

**File:** [`SSTCa2_population.py`](SSTCa2_population.py:543)  
**Location:** After `get_crossreg_data()` method (after line 580)  
**Priority:** HIGH - Core fix

### Description
Implement a new `z_normalize()` method in the `CrossregPCA` class that normalizes neural activity per neuron, per session using per-session statistics. This method will return a dict instead of a single array.

### Implementation Details
- Add method after line 580 (after `get_crossreg_data()`)
- Method should iterate over `self.S_crossreg.items()`
- For each session, z-score normalize each cell's activity using that session's mean/std
- Store original data in `self.S_original` dict
- Store normalized data in `self.S_normalized` dict
- Replace NaN with 0 using `np.nan_to_num()`
- Return `self.S_normalized` dict

### Testing Criteria
- Verify method returns a dict with keys matching session names from `self.crossreg_session_names`
- Verify each normalized array has same shape as original
- Verify normalized data contains no NaN values
- Verify normalized data has approximately zero mean and unit variance per cell (within each session)

---

## Subtask 2: Add `select_engram_cells()` Override for `CrossregPCA`

**File:** [`SSTCa2_population.py`](SSTCa2_population.py:543)  
**Location:** After Subtask 1 implementation  
**Priority:** HIGH - Required for engram cell selection

### Description
Override the inherited `select_engram_cells()` method to work with dict-based normalized data. The method should compute total activity across all sessions for each cell.

### Implementation Details
- Add method after Subtask 1
- Method should sum normalized activity across all sessions for each cell
- Use `scipy.stats.zscore()` to z-score the total activity
- Select cells where z-scored activity > threshold
- Store result in `self.engram_cells` as boolean array
- Add validation to check `self.S_normalized` is a dict

### Testing Criteria
- Verify method returns boolean array of shape `(n_cells,)`
- Verify `self.engram_cells` is set correctly
- Verify method raises ValueError if `z_normalize()` not called first
- Verify engram cell count is reasonable (not all or none)

---

## Subtask 3: Add `apply_smoothing()` Override for `CrossregPCA`

**File:** [`SSTCa2_population.py`](SSTCa2_population.py:543)  
**Location:** After Subtask 2 implementation  
**Priority:** MEDIUM - Required for smoothing pipeline

### Description
Override the inherited `apply_smoothing()` method to apply Gaussian smoothing along the time axis for each session independently.

### Implementation Details
- Add method after Subtask 2
- Method should iterate over `self.S_normalized.items()`
- Use `scipy.ndimage.gaussian_filter1d()` with `axis=1` for each session
- Store smoothed data in `self.S_smoothed` dict
- Add validation to check `self.S_normalized` is a dict

### Testing Criteria
- Verify method returns dict with same keys as `self.S_normalized`
- Verify smoothed data has same shape as input
- Verify smoothed data is less noisy (check variance reduction)
- Verify method raises ValueError if `z_normalize()` not called first

---

## Subtask 4: Update `__init__()` to Initialize New Instance Variables

**File:** [`SSTCa2_population.py`](SSTCa2_population.py:543)  
**Location:** In `CrossregPCA.__init__()` method (lines 543-555)  
**Priority:** MEDIUM - Required for proper state management

### Description
Add initialization for new instance variables that will be used by the new methods.

### Implementation Details
- Add `self.S_original = None` (will be dict)
- Add `self.S_normalized = None` (will be dict)
- Add `self.S_smoothed = None` (will be dict)
- These should be initialized in `__init__()` alongside existing `self.S_crossreg`

### Testing Criteria
- Verify all new attributes are initialized to None in `__init__()`
- Verify attributes are properly set after calling respective methods

---

## Subtask 5: Modify `fit_pca_method1()` to Avoid Double-Normalization

**File:** [`SSTCa2_population.py`](SSTCa2_population.py:582)  
**Location:** `fit_pca_method1()` method (lines 582-632)  
**Priority:** HIGH - Prevents incorrect normalization

### Description
Update `fit_pca_method1()` to check if `self.S_normalized` dict is available (from prior `z_normalize()` call). If available, use pre-normalized data directly. If not, fall back to internal normalization.

### Implementation Details
- Add check at start of method: if `self.S_normalized` is a dict, use it directly
- When using pre-normalized data, skip the internal normalization step (lines 604-608 for encoding, lines 621-624 for other sessions)
- Maintain backward compatibility: if `self.S_normalized` is None or not a dict, use existing internal normalization
- Ensure engram cell masking still works correctly with pre-normalized data

### Testing Criteria
- Verify PCA results are identical whether using pre-normalized or internally normalized data
- Verify method still works when `z_normalize()` is not called (backward compatibility)
- Verify PCA components are valid (non-zero, proper shape)

---

## Subtask 6: Modify `fit_pca_method2()` to Avoid Double-Normalization

**File:** [`SSTCa2_population.py`](SSTCa2_population.py:634)  
**Location:** `fit_pca_method2()` method (lines 634-684)  
**Priority:** HIGH - Prevents incorrect normalization

### Description
Update `fit_pca_method2()` to check if `self.S_normalized` dict is available (from prior `z_normalize()` call). If available, use pre-normalized data directly. If not, fall back to internal normalization.

### Implementation Details
- Add check at start of method: if `self.S_normalized` is a dict, use it directly
- When using pre-normalized data, skip the internal normalization step (lines 653-657 for concatenation, lines 672-675 for individual sessions)
- Maintain backward compatibility: if `self.S_normalized` is None or not a dict, use existing internal normalization
- Ensure engram cell masking still works correctly with pre-normalized data

### Testing Criteria
- Verify PCA results are identical whether using pre-normalized or internally normalized data
- Verify method still works when `z_normalize()` is not called (backward compatibility)
- Verify PCA components are valid (non-zero, proper shape)

---

## Subtask 7: Update `run_population_pca_pipeline()` to Work with New Structure

**File:** [`SSTCa2_population.py`](SSTCa2_population.py:1680)  
**Location:** Pipeline calls at lines 1688 and 1735  
**Priority:** HIGH - End-to-end integration

### Description
Verify that the pipeline calls in `run_population_pca_pipeline()` work correctly with the new dict-based structure. The existing call pattern should work:
```python
crossreg_pca.z_normalize()
crossreg_pca.select_engram_cells(threshold=engram_thresh)
crossreg_pca.apply_smoothing(sigma=smoothing_sigma)
crossreg_pca.fit_pca_method2(n_components=3)
```

### Implementation Details
- Review lines 1686-1718 (AVG mode) and lines 1733-1759 (FULL mode)
- Verify method call order is correct
- Verify no additional changes needed for plotting methods
- Check that `plot_crossreg_trajectories()` and related methods handle dict-based PCA results

### Testing Criteria
- Verify `run_population_pca_pipeline()` completes without errors for both AVG and FULL modes
- Verify PCA results are saved correctly
- Verify plots are generated correctly

---

## Subtask 8: Verify Plotting Methods Work with New Structure

**File:** [`SSTCa2_population.py`](SSTCa2_population.py)  
**Location:** Plotting methods in `CrossregPCA` class  
**Priority:** MEDIUM - Visualization integration

### Description
Verify that existing plotting methods in `CrossregPCA` work correctly with the new dict-based data structure. Check methods like `plot_crossreg_trajectories()`, `calculate_overlap()`, etc.

### Implementation Details
- Review all plotting methods in `CrossregPCA` class
- Verify they handle dict-based PCA results correctly
- Verify trajectory plotting works for all sessions
- Verify overlap calculations work correctly

### Testing Criteria
- Verify all plotting methods execute without errors
- Verify generated plots show correct data
- Verify plot labels and legends are correct

---

## Subtask 9: Add Unit Tests

**File:** New test file or existing test suite  
**Priority:** LOW - Verification

### Description
Add unit tests for all new/modified methods to ensure correctness and prevent regressions.

### Test Cases
1. `test_z_normalize_returns_dict` - Verify dict structure
2. `test_z_normalize_per_session_stats` - Verify per-session normalization
3. `test_select_engram_cells_returns_boolean_array` - Verify return type
4. `test_apply_smoothing_returns_dict` - Verify dict structure
5. `test_fit_pca_method1_with_pre_normalized` - Verify PCA with pre-normalized data
6. `test_fit_pca_method2_with_pre_normalized` - Verify PCA with pre-normalized data
7. `test_backward_compatibility_no_pre_normalized` - Verify methods work without pre-normalization
8. `test_pipeline_integration` - Verify full pipeline works end-to-end

---

## Execution Order

```
Subtask 4 (init variables)
    ↓
Subtask 1 (z_normalize method)
    ↓
Subtask 2 (select_engram_cells override)
    ↓
Subtask 3 (apply_smoothing override)
    ↓
Subtask 5 (fit_pca_method1 update)
    ↓
Subtask 6 (fit_pca_method2 update)
    ↓
Subtask 7 (pipeline integration)
    ↓
Subtask 8 (plotting verification)
    ↓
Subtask 9 (unit tests)
```

---

## Risk Assessment

| Subtask | Risk Level | Mitigation |
|---------|-----------|------------|
| 1 | Low | Straightforward implementation, well-defined |
| 2 | Low | Follows similar pattern to parent class |
| 3 | Low | Simple loop over dict items |
| 4 | Low | Simple variable initialization |
| 5 | Medium | Need to ensure backward compatibility |
| 6 | Medium | Need to ensure backward compatibility |
| 7 | Medium | Depends on all previous subtasks |
| 8 | Low | Mostly verification, minimal changes |
| 9 | Low | Standard unit tests |

---

## Files Modified

| File | Changes |
|------|---------|
| [`SSTCa2_population.py`](SSTCa2_population.py:543) | Add methods to `CrossregPCA` class, update `__init__`, modify PCA methods |

---

## Notes

- All new methods should include docstrings following the existing code style
- Use `np.nanmean` and `np.nanstd` for NaN-safe statistics
- Use `np.nan_to_num` to replace NaN with 0 after normalization
- Maintain backward compatibility - existing code that doesn't call `z_normalize()` should still work
- The dict-based structure (`self.S_normalized[name]`) replaces the single array (`self.S_normalized`)
