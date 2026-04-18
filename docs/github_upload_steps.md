# GitHub Upload Steps

Because GitHub CLI (`gh`) is not installed in this environment, repo creation needs one manual action in browser.

## 1. Create an empty repository on GitHub
- Name suggestion: `rl-agent-trading-research`
- Keep it empty (do not add README/.gitignore/license)

## 2. Connect local repo and push
Run in PowerShell:

```powershell
Set-Location "c:\Users\22130\Desktop\worksapce\projects\rl_agent_research"
git add .
git commit -m "feat: clean structure, add multi-seed suite and pdf reporting"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 3. If remote already exists
```powershell
git remote set-url origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 4. Verify
- Open the GitHub repo page.
- Confirm code under `src/`, configs under `configs/`, and report files under `docs/reports/`.
