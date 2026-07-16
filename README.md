# TMA4320 Prosjekt 1
1. prosjekt i TMA4320 Introduksjon til vitenskapelige beregninger, hvor det ble laget og reflektert rundt hvordan man kan bruke fysikk-informerte nevrale nettverk til å modellere varmeutvikling i et rom. Vi diskuterer blant annet i hvilken grad nettverket kan "lære" de fysiske parametrene inngående i varmeligningen og randbetingelsene, og ulike måter man kan lære nettverket på.


## Installer pakker i et virtuelt miljø

### pip og venv

Lag et virtuelt miljø

```bash
python -m venv .venv
```

Aktiver det virtuelle miljøet

```bash
. .venv\scripts\activate
```

Oppdater pip inne i det virtuelle miljøet

```bash
pip install --upgrade pip setuptools wheel
```

Installer pakkene i `pyproject.toml` med kommandoen

```bash
pip install -e . --group dev
```

Sett opp en kernel for Jupyter notebooks med

```bash
python -m ipykernel install --user --name project
```

## Kjøre kode

Pass på at det virtuelle miljøet er aktivert

```bash
. .venv\bin\activate
```

Kjør koden fra terminalen gjennom å skrive `python <filsti>`. Test at alt fungerer ved å kjøre `scripts/run_fdm.py` med kommandoen

```bash
python scripts/run_fdm.py
```

For å kjøre tester på koden kan du bruke `pytest`. Kjør alle tester med kommandoen

```bash
pytest
```
eller kjør en spesifikk testfil med kommandoen

```bash
pytest tests/<sti-til-test>.py
# eller
pytest tests/<sti-til-test>.py::<test-klassenavn>
```

