# TMA4320 Prosjekt 1
Denne teksten beskriver hvordan dere kan sette opp prosjektet, installere nødvendige pakker, og kjøre koden lokalt på deres maskin. Følg instruksjonene nøye for å sikre at alt fungerer som det skal.

## Laste ned koden

Følg instruksjonene i Git-forelesningen for å kopiere koden som en template og laste den ned til deres maskin. For de som ikke ønsker å bruke Git, kan dere laste ned en `zip`-fil med koden direkte fra GitHub og pakke den ut på deres maskin.


## Åpne prosjektet i en kode-editor


For å kjøre kommandoene nedenfor trenger dere også tilgang til en terminal. Det letteste er å åpne prosjektet i VSCode og bruke den innebygde terminalen (åpnes med `Ctrl`/`Cmd`+`j`). Eventuelt bruk terminalen på deres maskin direkte, men husk å navigere til mappen med koden først.

## Installere Python

Verifiser at dere har installert Python 3.13 eller nyere på deres maskin. Dette kan dere gjøre ved å åpne terminalen og skrive `python --version` eller `python3 --version`.

```bash
python --version
# eller evt. (avhengig av systemet deres)
python3 --version
```

Dersom dere får opp en versjon som er 3.13 eller nyere, er alt i orden. Hvis ikke, må dere installere Python. Følg instruksjonene på [Python Downloads](https://www.python.org/downloads/) for å installere nyeste versjon av Python på deres maskin.

> Merk: Dersom dere måtte skrive `python3` for å få opp riktig versjon, må dere bruke `python3` i stedet for `python` i alle kommandoer nedenfor`

## Installere pakker i et virtuelt miljø

For å kjøre koden trenger dere å installere noen Python-pakker og sette opp et virtuelt miljø. Fra tidligere er dere nok mest vant med å bruke pip og venv for å håndtere dette. Et mer moderne alternativ er å bruke [uv](https://uv.astral.sh/). Personlig synes jeg dette er et enklere og bedre verktøy for dette formålet. Jeg vil beskrive begge fremgangsmåtene, og dere kan selv velge hvilket dere ønsker å bruke.

### uv

Følg instruksjonene på [Installing uv](https://docs.astral.sh/uv/getting-started/installation/) for å sette opp på deres maskin. Verifiser at uv er installert ved å kjøre

```bash
uv --version
```

Dersom dere ønsker kan dere også enkelt installere nyeste version av Python gjennom uv ved å kjøre (merk at dette er valgfritt)

```bash
uv python install
```
Til slutt genererer du et virtuelt miljø og installerer pakker spesifisert i `pyproject.toml` ved å bruke kommandoen

```bash
uv sync
```

Sett opp en kernel for Jupyter notebooks gjennom

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
```

Dersom du ikke får noen feil har alt gått fint!

### pip og venv

Lag et virtuelt miljø

```bash
python -m venv .venv
```

Aktiver det virtuelle miljøet

```bash
source .venv/bin/activate
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

Dersom du ikke får noen feil har alt gått fint!

## Velge Interpreter i VSCode (dersom du bruker dette...)
Etter du har satt opp prosjektet kan det hende du må velge riktig Python Iinterpreter i VSCode. Åpne kommandopalletten med `Ctrl`/`Cmd`+`Shift`+`P`, og søk etter `Python: Select Interpreter`. Velg deretter den som peker til `.venv`-mappen i prosjektet ditt.

## Kjøre kode

Pass på at ditt virtuelle miljø er aktivert

```bash
source .venv/bin/activate
```

Vi kjører kode fra terminalen gjennom å skrive `python <filsti>`. Test at alt fungerer ved å kjøre `scripts/run_fdm.py` med kommandoen

```bash
python scripts/run_fdm.py
```

Dersom du ikke får noen feil så er alle pakker installert riktig, og du er klar til å begynne på prosjektet! Merk at output fra kjøringen vil selvfølgelig være gal siden vi ikke har implementert noe enda.

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

