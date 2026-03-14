# Safe Deletions After Reorganization

After running `REORGANIZATION.bat`, these files/folders can be safely deleted:

---

## ✅ SAFE TO DELETE (Auto-Generated / Build Artifacts)

| Item | Why Safe | Command |
|------|----------|---------|
| `.next/` | Build output, regenerated on `pnpm build` | `rmdir /s /q .next` |
| `frontend/.next/` | Build output in new location | `rmdir /s /q frontend\.next` |
| `node_modules/` | Reinstalled via `pnpm install` | `rmdir /s /q node_modules` |
| `frontend/node_modules/` | Reinstalled via `pnpm install` in frontend | `rmdir /s /q frontend\node_modules` |
| `.venv/` | Python virtual env (if exists) | `rmdir /s /q .venv` |
| `backend/.venv/` | Backend virtual env (if exists) | `rmdir /s /q backend\.venv` |
| `__pycache__/` (all) | Python bytecode cache | (auto-recreated, safe to delete) |
| `*.pyc` files | Python compiled files | (auto-recreated, safe to delete) |
| `.pytest_cache/` | Test cache | `rmdir /s /q .pytest_cache` |
| `*.egg-info/` | Python package info | (auto-recreated, safe to delete) |

---

## ⚠️ OPTIONAL / CONDITIONAL DELETIONS

### If No Longer Needed (Optionally Delete)

| Item | Reason | Decision |
|------|--------|----------|
| `docs/DEPLOYMENT_COMPLETE.md` | Old deployment status (information superseded) | ✓ Can delete if using new process |
| `docs/FABRICSENSE_AI_AUDIT_REPORT.md` | Audit snapshot (new audits can regenerate) | ✓ Can delete if archived elsewhere |
| `docs/PREDICTION_FIX_SUMMARY.md` | Historical fix notes (no longer relevant) | ✓ Can delete if not needed for reference |
| `docs/Readme.txt` | Duplicate of markdown docs | ✓ Should delete |

### Recommended to Keep

| Item | Reason | Keep |
|------|--------|------|
| `docs/README.md` | Project overview | ✓ Keep |
| `frontend/` | Entire folder | ✓ Keep |
| `backend/` | Entire folder | ✓ Keep |
| `ml-pipeline/` | Entire folder | ✓ Keep |
| `.vscode/` | IDE settings | ✓ Keep |
| `.gitignore` | Git configuration | ✓ Keep |
| Root `README.md` (new) | Project documentation | ✓ Keep |

---

## Batch Commands to Clean Up

### Delete All Auto-Generated Files
```batch
@echo off
REM Delete build artifacts and caches
rmdir /s /q .next 2>nul
rmdir /s /q frontend\.next 2>nul
rmdir /s /q node_modules 2>nul
rmdir /s /q frontend\node_modules 2>nul
rmdir /s /q __pycache__ 2>nul
rmdir /s /q .pytest_cache 2>nul

REM Delete Python caches
for /r . %%d in (__pycache__) do @rmdir /s /q "%%d" 2>nul
for /r . %%f in (*.pyc) do @del "%%f" 2>nul

echo Cleanup complete!
pause
```

### Delete Only Non-Critical Docs (Optional)
```batch
@echo off
del "docs\DEPLOYMENT_COMPLETE.md" 2>nul
del "docs\FABRICSENSE_AI_AUDIT_REPORT.md" 2>nul
del "docs\PREDICTION_FIX_SUMMARY.md" 2>nul
del "docs\Readme.txt" 2>nul

echo Optional docs removed!
pause
```

---

## What NOT to Delete

❌ **NEVER DELETE:**
- Any file in `frontend/app/` or `frontend/components/`
- `backend/fabric_backend/main.py` and `requirements.txt`
- `ml-pipeline/model/textile_classifier_rf.pkl`
- `.gitignore` and `.git/` directory
- Root `README.md`
- `frontend/package.json` and `package.json` files

---

## Disk Space Savings

Deleting auto-generated files will save approximately:

| Item | Typical Size | Deleted |
|------|--------------|---------|
| `.next/` | 100-300 MB | ✓ Can rebuild on demand |
| `node_modules/` | 300-500 MB | ✓ Reinstall with `pnpm install` |
| `__pycache__/` | 5-50 MB | ✓ Auto-recreated |

**Total potential savings: 400-850 MB** (depending on dependencies)

---

## Reinstalling After Deletion

If you delete `node_modules/` or `__pycache__/`:

### Frontend
```bash
cd frontend
pnpm install
pnpm dev
```

### Backend
```bash
cd backend/fabric_backend
pip install -r requirements.txt
python main.py
```

---

## Summary

**After running `REORGANIZATION.bat`:**

1. ✓ Update required files (see `FILE_UPDATES_REQUIRED.md`)
2. ✓ Replace root README with `README_NEW.md`
3. ⚠️ Optional: Delete old audit/deployment docs
4. ✓ Delete auto-generated folders (`.next/`, `node_modules/`)
5. ✓ Verify project runs with quick test
6. ✓ Commit to git with cleaned structure
