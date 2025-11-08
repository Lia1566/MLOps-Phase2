# GitHub Actions Workflows

## CI/CD Pipelines

### 1. **CI Pipeline** (`ci.yml`)
Runs on every push and pull request.

**Jobs:**
- **Test**: Runs unit tests on Python 3.10 and 3.11
- **Lint**: Code quality checks (Black, isort, flake8)
- **Security**: Dependency vulnerability scanning
- **Integration**: API integration tests

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`
- Manual trigger

---

### 2. **Docker Build** (`docker.yml`)
Builds and tests Docker images.

**Jobs:**
- Builds Docker image
- Tests image health
- (Optional) Pushes to Docker Hub

**Triggers:**
- Push to `main`
- Version tags (`v*`)
- Manual trigger

---

### 3. **Drift Monitoring** (`drift-check.yml`)
Scheduled data drift monitoring.

**Jobs:**
- Runs drift detection tests
- Logs drift monitoring results

**Triggers:**
- Daily at 9 AM UTC
- Manual trigger

---

## Status Badges

Add to your main README.md:
```markdown
![CI Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/CI%20Pipeline/badge.svg)
![Docker Build](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Docker%20Build%20%26%20Push/badge.svg)
```

---

## Setup Instructions

### 1. Push to GitHub
```bash
git add .
git commit -m "Add GitHub Actions CI/CD pipelines"
git push origin main
```

### 2. Check Actions Tab
Navigate to your repository → **Actions** tab to see workflows running

### 3. (Optional) Docker Hub Setup
To enable Docker Hub push:
1. Create Docker Hub account
2. Generate access token
3. Add secrets to GitHub:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
4. Uncomment push section in `docker.yml`

---

## Monitoring

- View workflow runs in GitHub Actions tab
- Check logs for detailed output
- Failed jobs will show in PR checks
- Email notifications for failures (configure in settings)