# GitHub Actions CI Fixes Summary

## Overview
Fixed failing GitHub Actions tests. **All 13 tests now passing** (100% success rate)!

## Issues Fixed

### 1. Napari Plugin Manifest Naming Mismatch ✅
**File:** `src/adc/napari.yaml`

**Problem:** Package name used underscores (`anchor_droplet_chip`) instead of hyphens (`anchor-droplet-chip`), causing plugin registration to fail.

**Fix:**
- Changed line 1: `name: anchor-droplet-chip`
- Updated all command IDs to use hyphens instead of underscores throughout the file

### 2. Missing Test Dependency (h5py) ✅
**File:** `tox.ini`

**Problem:** Tests were failing because `h5py` module was not installed.

**Fix:** Added `h5py` to the test dependencies (line 38)

### 3. Path Handling Bugs ✅
**File:** `src/adc/_split_stack.py`

**Problems:**
- Code crashed when `self.path` was None (when layers are created programmatically without files)
- `update_table()` didn't regenerate filenames when `path_widget` changed
- Directory paths without placeholders weren't handled correctly

**Fixes:**
- Lines 182: Store `letter` for use in `update_table()`
- Lines 199-204: Added null check before using `self.path`
- Lines 213-240: Made `update_table()` regenerate names from current `path_widget.value`
- Lines 218-230: Handle both placeholder patterns and plain directory paths
- Lines 269-274: Added proper exception handling for missing path in layer metadata

## Dev Container Configuration ✅
**File:** `.devcontainer/devcontainer.json`

**Added:**
- System dependencies for OpenGL and Qt6
- Automated installation of xvfb for headless testing
- Installation of h5py and test dependencies

## GitHub Actions Local Testing ✅
**Tool:** `act` installed in container

Can now run workflows locally:
```bash
act -l  # List workflows
```

## Test Results

### ✅ All Tests Passing (13/13) - 100% Success Rate!

- ✅ test_to_8bits
- ✅ test_count
- ✅ test_crop
- ✅ test_count2d
- ✅ test_recursive
- ✅ test_projection
- ✅ test_read_zarr
- ✅ test_read_small_tif
- ✅ test_read_big_tif
- ✅ test_split_to_layers
- ✅ test_substack
- ✅ test_projection (stack)
- ✅ test_substack_single_channel

## How to Run Tests

### In Dev Container:
```bash
# With xvfb (headless)
xvfb-run -a pytest -v --color=yes

# With tox
tox
```

### Locally with act:
```bash
act -j test  # Run the test job
```

## Files Modified
1. `src/adc/napari.yaml` - Fixed plugin manifest
2. `tox.ini` - Added h5py dependency
3. `src/adc/_split_stack.py` - Fixed path handling
4. `.devcontainer/devcontainer.json` - Added system dependencies
5. `.devcontainer/devcontainer.env` - Environment variables for Qt

## Next Steps (Optional)
- Consider adding act to devcontainer features for easier workflow testing
- Add pre-commit hooks to validate napari manifest
- Set up code coverage reporting
