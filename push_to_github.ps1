# Script PowerShell — initialise le dépôt et pousse sur GitHub
# Utilisation :
#   .\push_to_github.ps1 -Repo "concrete-optimizer" -User "votre-username-github"

param(
    [Parameter(Mandatory=$true)]
    [string]$User,

    [Parameter(Mandatory=$false)]
    [string]$Repo = "concrete-optimizer"
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== Initialisation du dépôt Git ===" -ForegroundColor Cyan

git init
git add .
git commit -m "feat: concrete mix optimiser + durability simulation

Multi-objective optimisation (NSGA-II) for concrete formulation:
- Minimise embodied CO2 (IPCC/Ecoinvent factors)
- Maximise compressive strength (Abrams law + k-values)
- Maximise service life (carbonation + chloride models)

Scientific models implemented:
- Carbonation: Papadakis (1991) / fib Bulletin 34
- Chloride ingress: Fick 2nd law / DuraCrete (2000)
- Corrosion: Rodriguez (1996) / Liu & Weyers (1998)
- NSGA-II: Deb et al. (2002) with SBX + polynomial mutation

Interactive Dash dashboard + matplotlib publication figures."

Write-Host "`n=== Connexion au dépôt distant ===" -ForegroundColor Cyan
Write-Host "Assurez-vous d'avoir créé le dépôt '$Repo' sur https://github.com/$User/$Repo" -ForegroundColor Yellow
Write-Host "Appuyez sur Entrée quand c'est fait..."
Read-Host

git branch -M main
git remote add origin "https://github.com/$User/$Repo.git"
git push -u origin main

Write-Host "`n=== Dépôt publié ===" -ForegroundColor Green
Write-Host "URL : https://github.com/$User/$Repo" -ForegroundColor Green
