\
from pathlib import Path
from datetime import date
import requests

DFP = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
ITR = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_{year}.zip"

def download_file(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[SKIP] {dest.name}")
        return
    print(f"[GET] {url}")
    with requests.get(url, timeout=120, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

def download_cvm(start_year=2012, end_year=None):
    end_year = end_year or date.today().year
    base = Path("data/cvm/raw")
    for year in range(start_year, end_year + 1):
        for kind, template in (("dfp", DFP), ("itr", ITR)):
            url = template.format(year=year)
            dest = base / kind / Path(url).name
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"[WARN] {kind.upper()} {year}: {e}")
