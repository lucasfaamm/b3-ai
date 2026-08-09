
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import re

import numpy as np
import pandas as pd
import requests
import yaml

from src.brapi_client import BrapiClient

CVM_DFP = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"


def digits(x):
    return re.sub(r"\D", "", str(x or ""))


def num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def pct_growth(cur, prev):
    try:
        cur = float(cur)
        prev = float(prev)
        if not np.isfinite(cur) or not np.isfinite(prev) or prev == 0:
            return np.nan
        return (cur - prev) / abs(prev)
    except Exception:
        return np.nan


def download_zip(url):
    print(f"[CVM] baixando {url}", flush=True)
    r = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "b3-ai-personal-radar/1.1"},
    )
    r.raise_for_status()
    return ZipFile(BytesIO(r.content))


def read_member(zf, needle):
    names = [n for n in zf.namelist() if needle.lower() in n.lower() and n.lower().endswith(".csv")]
    if not names:
        return pd.DataFrame()
    with zf.open(names[0]) as fh:
        return pd.read_csv(fh, sep=";", encoding="latin1", low_memory=False)


def prepare_statement(df):
    if df.empty:
        return df

    x = df.copy()

    if "CNPJ_CIA" in x.columns:
        x["CNPJ_NORM"] = x["CNPJ_CIA"].map(digits)

    if "ORDEM_EXERC" in x.columns:
        order = x["ORDEM_EXERC"].astype(str).str.upper()
        last = order.str.contains("ÚLTIMO|ULTIMO", regex=True, na=False)
        if last.any():
            x = x[last].copy()

    if "VL_CONTA" in x.columns:
        s = x["VL_CONTA"].astype(str).str.strip()
        comma = s.str.contains(",", regex=False)
        s.loc[comma] = (
            s.loc[comma]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        x["VL_NUM"] = pd.to_numeric(s, errors="coerce")

        # CRÍTICO: normaliza todos os demonstrativos para REAIS.
        # A CVM frequentemente publica VL_CONTA com ESCALA_MOEDA="MIL".
        if "ESCALA_MOEDA" in x.columns:
            escala = x["ESCALA_MOEDA"].astype(str).str.upper().str.strip()
            is_mil = escala.str.contains(r"\bMIL\b", regex=True, na=False)
            x.loc[is_mil, "VL_NUM"] = x.loc[is_mil, "VL_NUM"] * 1000.0

    sort_cols = [c for c in ["CNPJ_NORM", "DT_REFER", "CD_CONTA", "VERSAO"] if c in x.columns]
    if sort_cols:
        x = x.sort_values(sort_cols)
        keys = [c for c in ["CNPJ_NORM", "DT_REFER", "CD_CONTA"] if c in x.columns]
        if keys:
            x = x.drop_duplicates(keys, keep="last")

    return x


def combine_con_ind(zf, base, year):
    con = prepare_statement(read_member(zf, f"_{base}_con_{year}"))
    ind = prepare_statement(read_member(zf, f"_{base}_ind_{year}"))

    if con.empty:
        return ind
    if ind.empty:
        return con

    have_con = set(con.get("CNPJ_NORM", pd.Series(dtype=str)).dropna())
    ind_only = ind[~ind["CNPJ_NORM"].isin(have_con)].copy()
    return pd.concat([con, ind_only], ignore_index=True)


def account_map(df, code):
    if df.empty or "CD_CONTA" not in df.columns:
        return {}
    z = df[df["CD_CONTA"].astype(str).eq(str(code))].copy()
    if z.empty:
        return {}
    return (
        z.dropna(subset=["CNPJ_NORM"])
        .set_index("CNPJ_NORM")["VL_NUM"]
        .to_dict()
    )


def load_dfp_year(year):
    zf = download_zip(CVM_DFP.format(year=year))

    dre = combine_con_ind(zf, "DRE", year)
    bpa = combine_con_ind(zf, "BPA", year)
    bpp = combine_con_ind(zf, "BPP", year)

    revenue = account_map(dre, "3.01")
    op_income = account_map(dre, "3.05")

    net_parent = account_map(dre, "3.11.01")
    net_total = account_map(dre, "3.11")

    equity = account_map(bpp, "2.03")
    cash = account_map(bpa, "1.01.01")

    debt_cur = account_map(bpp, "2.01.04")
    debt_noncur = account_map(bpp, "2.02.01")

    cnpjs = set()
    for m in (revenue, op_income, net_parent, net_total, equity, cash, debt_cur, debt_noncur):
        cnpjs.update(m.keys())

    out = {}
    for cnpj in cnpjs:
        ni = net_parent.get(cnpj, np.nan)
        if pd.isna(ni):
            ni = net_total.get(cnpj, np.nan)

        d1 = debt_cur.get(cnpj, 0.0)
        d2 = debt_noncur.get(cnpj, 0.0)
        debt = (0 if pd.isna(d1) else d1) + (0 if pd.isna(d2) else d2)

        out[cnpj] = {
            "revenue": revenue.get(cnpj, np.nan),
            "operating_income": op_income.get(cnpj, np.nan),
            "net_income": ni,
            "equity": equity.get(cnpj, np.nan),
            "cash": cash.get(cnpj, np.nan),
            "debt": debt,
        }
    return out


with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

limit = int(cfg["runtime"].get("max_fundamental_tickers", 300))
workers = int(cfg["runtime"].get("fundamental_workers", 4))
pause = float(cfg["runtime"].get("request_pause_seconds", 0.10))

client = BrapiClient(pause=pause, max_retries=4)
items = client.tickers(limit=2000)

universe = []
for item in items:
    if not isinstance(item, dict):
        continue

    symbol = str(item.get("symbol", "")).upper().strip()
    if not symbol or not item.get("isActive", True):
        continue
    if str(item.get("assetType", "")).lower() != "stock":
        continue
    if str(item.get("subType", "")).lower() not in ("", "stock", "unit"):
        continue

    q = item.get("quote") or {}
    universe.append({
        "ticker": symbol,
        "name": item.get("name") or item.get("longName") or symbol,
        "sector_catalog": item.get("sector") or "Unknown",
        "subsector_catalog": item.get("subsector") or "Unknown",
        "catalog_price": num(q.get("lastPrice")),
        "catalog_volume": num(q.get("volume")),
        "market_cap": num(q.get("marketCap")),
    })

universe = universe[:limit]

if not universe:
    raise SystemExit("Não foi possível montar o universo B3.")


def profile_one(row):
    c = BrapiClient(pause=pause, max_retries=3)
    p = c.profile(row["ticker"])
    return {
        **row,
        "cnpj": digits(p.get("cnpj")),
        "sector": p.get("sector") or p.get("sectorDisp") or row["sector_catalog"],
        "industry": p.get("industry") or p.get("industryDisp") or row["subsector_catalog"],
    }


profiles = []
profile_errors = []

with ThreadPoolExecutor(max_workers=workers) as ex:
    futures = {ex.submit(profile_one, r): r["ticker"] for r in universe}
    for i, fut in enumerate(as_completed(futures), start=1):
        ticker = futures[fut]
        try:
            profiles.append(fut.result())
            print(f"[PROFILE {i}/{len(universe)}] OK {ticker}", flush=True)
        except Exception as e:
            profile_errors.append({"ticker": ticker, "error": str(e)})
            print(f"[PROFILE {i}/{len(universe)}] ERRO {ticker}: {e}", flush=True)


current_year = int(pd.Timestamp.utcnow().year) - 1
previous_year = current_year - 1

cur = load_dfp_year(current_year)
prev = load_dfp_year(previous_year)

rows = []
unmatched = []

for r in profiles:
    cnpj = r.get("cnpj", "")
    a = cur.get(cnpj)
    b = prev.get(cnpj)

    if not cnpj or a is None:
        unmatched.append({"ticker": r["ticker"], "cnpj": cnpj, "reason": "sem DFP/CNPJ match"})
        continue

    market_cap = r.get("market_cap", np.nan)
    revenue = a.get("revenue", np.nan)
    op_income = a.get("operating_income", np.nan)
    net_income = a.get("net_income", np.nan)
    equity = a.get("equity", np.nan)
    cash = a.get("cash", np.nan)
    debt = a.get("debt", np.nan)

    prev_rev = b.get("revenue", np.nan) if b else np.nan
    prev_ni = b.get("net_income", np.nan) if b else np.nan

    net_debt = debt - cash if not pd.isna(debt) and not pd.isna(cash) else np.nan
    enterprise_value = (
        market_cap + net_debt
        if not pd.isna(market_cap) and not pd.isna(net_debt)
        else np.nan
    )

    earnings_yield = safe_div(net_income, market_cap)
    book_to_price = safe_div(equity, market_cap)
    sales_yield = safe_div(revenue, market_cap)
    ev_to_ebit = (
        safe_div(enterprise_value, op_income)
        if not pd.isna(op_income) and op_income > 0
        else np.nan
    )

    pe = safe_div(1.0, earnings_yield) if not pd.isna(earnings_yield) and earnings_yield > 0 else np.nan
    pb = safe_div(1.0, book_to_price) if not pd.isna(book_to_price) and book_to_price > 0 else np.nan

    rows.append({
        "ticker": r["ticker"],
        "cnpj": cnpj,
        "company_name": r["name"],
        "sector": r["sector"],
        "industry": r["industry"],
        "market_cap": market_cap,
        "revenue": revenue,
        "operating_income": op_income,
        "net_income": net_income,
        "equity": equity,
        "total_cash": cash,
        "total_debt": debt,
        "roe": safe_div(net_income, equity),
        "operating_margin": safe_div(op_income, revenue),
        "profit_margin": safe_div(net_income, revenue),
        "debt_to_equity": safe_div(debt, equity),
        "revenue_growth": pct_growth(revenue, prev_rev),
        "earnings_growth": pct_growth(net_income, prev_ni),
        "earnings_yield": earnings_yield,
        "book_to_price": book_to_price,
        "sales_yield": sales_yield,
        "ev_to_ebit": ev_to_ebit,
        "raw_pe": pe,
        "raw_pb": pb,
        "raw_ev_ebitda": ev_to_ebit,
        "enterprise_value": enterprise_value,
        "ebitda": op_income,
        "fcf": np.nan,
        "beta": np.nan,
        "trailing_eps": np.nan,
        "book_value_per_share": np.nan,
        "dfp_year": current_year,
        "previous_dfp_year": previous_year,
    })

fund = pd.DataFrame(rows)

Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

fund.to_csv("data/fundamentals.csv", index=False)
pd.DataFrame(profile_errors).to_csv("results/profile_errors.csv", index=False)
pd.DataFrame(unmatched).to_csv("results/cvm_unmatched.csv", index=False)

status = {
    "universe_requested": len(universe),
    "profiles_ok": len(profiles),
    "profile_errors": len(profile_errors),
    "cvm_matched": len(fund),
    "cvm_unmatched": len(unmatched),
    "dfp_year": current_year,
    "previous_dfp_year": previous_year,
    "financial_units": "BRL",
    "fundamental_source": "CVM DFP bulk",
}

Path("results/fundamentals_status.json").write_text(
    json.dumps(status, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

minimum_ok = max(20, int(len(universe) * 0.50))

print(f"[OK] fundamentos CVM: {len(fund)}/{len(universe)}", flush=True)

if len(fund) < minimum_ok:
    raise SystemExit(
        f"Cruzamento CVM insuficiente ({len(fund)}/{len(universe)})."
    )
