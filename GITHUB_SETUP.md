# GitHub Setup

## 1. Create repository
Create a new GitHub repository, for example:

```text
index-return-prediction-portfolio
```

## 2. Push local files

```bash
git init
git add .
git commit -m "Initial portfolio repository"
git branch -M main
git remote add origin https://github.com/<your-username>/index-return-prediction-portfolio.git
git push -u origin main
```

## 3. Enable GitHub Pages

1. Open repository **Settings**.
2. Go to **Pages**.
3. Source: **Deploy from a branch**.
4. Branch: `main`.
5. Folder: `/docs`.
6. Save.

Expected portfolio URL:

```text
https://<your-username>.github.io/index-return-prediction-portfolio/
```
