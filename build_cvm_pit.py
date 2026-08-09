
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json, os, re, subprocess, tempfile
import numpy as np
import pandas as pd

from src.brapi_client import BrapiClient

START_YEAR=2020
END_YEAR=pd.Timestamp.utcnow().year
OUT=Path("data"); OUT.mkdir(exist_ok=True)
RES=Path("results"); RES.mkdir(exist_ok=True)

def digits(x): return re.sub(r"\D","",str(x or ""))

# ---------- ticker -> CNPJ ----------
fund=pd.read_csv("data/fundamentals.csv")
tickers=fund["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()

def one_profile(t):
    c=BrapiClient(pause=.05,max_retries=4)
    p=c.profile(t)
    return {
        "ticker":t,
        "cnpj":digits(p.get("cnpj")),
        "sector":p.get("sector") or p.get("sectorDisp") or "Unknown",
        "industry":p.get("industry") or p.get("industryDisp") or "Unknown",
    }

maps=[];errs=[]
with ThreadPoolExecutor(max_workers=4) as ex:
    futs={ex.submit(one_profile,t):t for t in tickers}
    for f in as_completed(futs):
        t=futs[f]
        try: maps.append(f.result())
        except Exception as e: errs.append({"ticker":t,"error":str(e)})

mapping=pd.DataFrame(maps)
mapping=mapping[mapping["cnpj"].astype(str).str.len()>=8].drop_duplicates("ticker")
mapping.to_csv(OUT/"ticker_cnpj.csv",index=False)
pd.DataFrame(errs).to_csv(RES/"ticker_cnpj_errors.csv",index=False)
cnpjs=set(mapping["cnpj"])
if len(cnpjs)<30:
    raise SystemExit(f"Poucos CNPJs mapeados: {len(cnpjs)}")

# ---------- CVM downloader: optional Cloudflare bridge ----------
CVM_PROXY_BASE=os.getenv("CVM_PROXY_BASE","").strip().rstrip("/")

def download_zip(doc,year):
    doc=doc.upper()
    if CVM_PROXY_BASE:
        url=f"{CVM_PROXY_BASE}/{doc}/{year}.zip"
        print(f"[CVM bridge] {doc} {year}",flush=True)
    else:
        url=f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc}/DADOS/{doc.lower()}_cia_aberta_{year}.zip"
        print(f"[CVM direct] {doc} {year}",flush=True)
    with tempfile.NamedTemporaryFile(suffix=".zip",delete=False) as tmp:
        path=tmp.name
    cmd=[
        "curl","-L","--fail","--silent","--show-error",
        "--retry","5","--retry-all-errors","--retry-delay","3",
        "--connect-timeout","30","--max-time","300",
        "-o",path,url
    ]
    try:
        subprocess.run(cmd,check=True)
        if Path(path).stat().st_size < 1000:
            raise RuntimeError(f"Arquivo muito pequeno: {url}")
        b=Path(path).read_bytes()
        z=ZipFile(BytesIO(b))
        if z.testzip() is not None:
            raise RuntimeError(f"ZIP corrompido: {url}")
        return z
    finally:
        try: os.remove(path)
        except OSError: pass

def read_member(zf,contains):
    names=[n for n in zf.namelist() if contains.lower() in n.lower() and n.lower().endswith(".csv")]
    if not names:return pd.DataFrame()
    with zf.open(names[0]) as fh:
        return pd.read_csv(fh,sep=";",encoding="latin1",low_memory=False)

def norm(df):
    if df.empty:return df
    x=df.copy()
    if "CNPJ_CIA" in x.columns:x["cnpj"]=x["CNPJ_CIA"].map(digits)
    if "DT_REFER" in x.columns:x["reference_date"]=pd.to_datetime(x["DT_REFER"],errors="coerce")
    if "DT_RECEB" in x.columns:x["received_date"]=pd.to_datetime(x["DT_RECEB"],errors="coerce")
    if "VERSAO" in x.columns:x["version"]=pd.to_numeric(x["VERSAO"],errors="coerce").fillna(0).astype(int)
    else:x["version"]=0
    if "VL_CONTA" in x.columns:
        s=x["VL_CONTA"].astype(str).str.strip()
        comma=s.str.contains(",",regex=False)
        s.loc[comma]=s.loc[comma].str.replace(".","",regex=False).str.replace(",",".",regex=False)
        x["value"]=pd.to_numeric(s,errors="coerce")
        if "ESCALA_MOEDA" in x.columns:
            mil=x["ESCALA_MOEDA"].astype(str).str.upper().str.contains(r"\bMIL\b",regex=True,na=False)
            x.loc[mil,"value"]*=1000.0
    if "ORDEM_EXERC" in x.columns:
        last=x["ORDEM_EXERC"].astype(str).str.upper().str.contains("ÚLTIMO|ULTIMO",regex=True,na=False)
        if last.any():x=x[last].copy()
    if "cnpj" in x.columns:x=x[x["cnpj"].isin(cnpjs)].copy()
    return x

def general_map(zf,doc,year):
    g=norm(read_member(zf,f"{doc.lower()}_cia_aberta_{year}.csv"))
    if g.empty:return pd.DataFrame()
    cols=[c for c in ["cnpj","reference_date","version","received_date"] if c in g.columns]
    if "received_date" not in cols:return pd.DataFrame()
    return g[cols].dropna(subset=["cnpj","reference_date"]).drop_duplicates(["cnpj","reference_date","version"],keep="last")

def combine(zf,doc,statement,year):
    con=norm(read_member(zf,f"_{statement}_con_{year}"))
    ind=norm(read_member(zf,f"_{statement}_ind_{year}"))
    if con.empty:return ind
    if ind.empty:return con
    keys=["cnpj","reference_date","version"]
    have=set(map(tuple,con[keys].drop_duplicates().values.tolist()))
    mask=[tuple(x) not in have for x in ind[keys].values.tolist()]
    return pd.concat([con,ind.loc[mask]],ignore_index=True)

def acct(df,code):
    if df.empty or "CD_CONTA" not in df.columns:return pd.DataFrame()
    z=df[df["CD_CONTA"].astype(str).eq(code)].copy()
    return z[["cnpj","reference_date","version","value"]].rename(columns={"value":code})

def shares_table(zf,doc,year):
    x=norm(read_member(zf,"composicao_capital"))
    if x.empty:return pd.DataFrame()
    cand=["QT_ACAO_TOTAL_CAP_INTEGR","QT_ACAO_ORDIN_CAP_INTEGR","QT_ACAO_PREFER_CAP_INTEGR"]
    total=None
    if "QT_ACAO_TOTAL_CAP_INTEGR" in x.columns:
        total=pd.to_numeric(x["QT_ACAO_TOTAL_CAP_INTEGR"],errors="coerce")
    else:
        total=pd.Series(0.0,index=x.index)
        found=False
        for c in cand[1:]:
            if c in x.columns:
                total=total+pd.to_numeric(x[c],errors="coerce").fillna(0); found=True
        if not found:return pd.DataFrame()
    x["shares_outstanding"]=total
    return x[["cnpj","reference_date","version","shares_outstanding"]].drop_duplicates(
        ["cnpj","reference_date","version"],keep="last"
    )

rows=[]
for doc in ["DFP","ITR"]:
    for year in range(START_YEAR,END_YEAR+1):
        print(f"[CVM] {doc} {year}",flush=True)
        try:
            zf=download_zip(doc,year)
        except Exception as e:
            print(f"[WARN] download {doc} {year}: {e}",flush=True)
            continue

        gm=general_map(zf,doc,year)
        dre=combine(zf,doc,"DRE",year)
        bpa=combine(zf,doc,"BPA",year)
        bpp=combine(zf,doc,"BPP",year)
        shr=shares_table(zf,doc,year)

        keys=["cnpj","reference_date","version"]
        parts=[
            acct(dre,"3.01"),acct(dre,"3.05"),acct(dre,"3.11.01"),acct(dre,"3.11"),
            acct(bpp,"2.03"),acct(bpa,"1.01.01"),acct(bpp,"2.01.04"),acct(bpp,"2.02.01")
        ]
        wide=None
        for p in parts:
            if p.empty:continue
            wide=p if wide is None else wide.merge(p,on=keys,how="outer")
        if wide is None:continue

        if not gm.empty:wide=wide.merge(gm,on=keys,how="left")
        if not shr.empty:wide=wide.merge(shr,on=keys,how="left")

        # Latest received version for a particular reference date.
        if "received_date" not in wide.columns:
            lag=120 if doc=="DFP" else 60
            wide["received_date"]=wide["reference_date"]+pd.to_timedelta(lag,unit="D")
        else:
            lag=120 if doc=="DFP" else 60
            wide["received_date"]=wide["received_date"].fillna(
                wide["reference_date"]+pd.to_timedelta(lag,unit="D")
            )

        wide["document_type"]=doc
        wide["quarter"]=wide["reference_date"].dt.month.map({3:1,6:2,9:3,12:4})
        wide["net_income"]=wide.get("3.11.01")
        if "3.11" in wide.columns:
            wide["net_income"]=wide["net_income"].fillna(wide["3.11"])
        wide["revenue"]=wide.get("3.01")
        wide["operating_income"]=wide.get("3.05")
        wide["equity"]=wide.get("2.03")
        wide["cash"]=wide.get("1.01.01")
        wide["debt"]=wide.get("2.01.04",0).fillna(0)+wide.get("2.02.01",0).fillna(0)

        keep=["cnpj","reference_date","received_date","version","document_type","quarter",
              "revenue","operating_income","net_income","equity","cash","debt","shares_outstanding"]
        keep=[c for c in keep if c in wide.columns]
        rows.append(wide[keep])

if not rows:
    raise SystemExit(
        "Nenhum arquivo CVM foi obtido. Confira CVM_PROXY_BASE e o Worker."
    )

pit=pd.concat(rows,ignore_index=True)
pit=pit.dropna(subset=["cnpj","reference_date","received_date"])
pit=pit.sort_values(["cnpj","reference_date","received_date","version"])
pit=pit.drop_duplicates(["cnpj","reference_date","received_date"],keep="last")

# IMPORTANT: ITR DRE values are YTD/cumulative. Keep them RAW here.
# The next backtest will de-accumulate quarters and construct TTM using
# only filings that were already received at each historical rebalance date.
pit=pit.rename(columns={
    "revenue":"revenue_ytd_or_annual",
    "operating_income":"operating_income_ytd_or_annual",
    "net_income":"net_income_ytd_or_annual",
})

# Attach tickers. Multiple share classes can map to the same CNPJ.
pit=pit.merge(mapping,on="cnpj",how="inner")
pit.to_csv(OUT/"cvm_pit_fundamentals.csv.gz",index=False,compression="gzip")

status={
    "rows":int(len(pit)),
    "tickers":int(pit["ticker"].nunique()),
    "companies":int(pit["cnpj"].nunique()),
    "start_reference":str(pit["reference_date"].min().date()),
    "end_reference":str(pit["reference_date"].max().date()),
    "network_route":"cloudflare_worker" if CVM_PROXY_BASE else "direct_cvm",
    "source":"Official CVM DFP/ITR bulk files; received_date preserved.",
    "lookahead_control":"Only filings with received_date <= rebalance date may be used.",
    "itr_policy":"ITR flow values are kept cumulative/YTD here; do NOT annualize by 4/q.",
    "next_step":"Use corrected de-accumulation + TTM backtest; do not run the old research_final workflow."
}
(RES/"cvm_pit_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(status,ensure_ascii=False,indent=2))
