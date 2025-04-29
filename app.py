# ─────────────────────────────────────────────────────────────────────────────
# app.py — Bankruptcy Risk Predictor (Streamlit)
# 2025-04-25
#
# 1) 移除了折叠，改为直接文本的 Model 介绍与比率释义
# 2) 在财务比率图表中增加更宽的画布及 labelOverlap="greedy" 来避免文字重叠
# 3) 保留旧字段名映射 (X1_WC_TA 等) 兼容 ratio_stats.json
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import re, unicodedata
import joblib
import json
import altair as alt
from tensorflow.keras.models import load_model
from pathlib import Path

# ------------------------------------------------------------------------------
# 1) 页面配置
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Risk Predictor",
    page_icon="📉",
    layout="wide"
)

# ------------------------------------------------------------------------------
# 2) 文件路径常量
# ------------------------------------------------------------------------------
EXTRA_INFO_CSV = Path("df_text_final.csv")
STATS_JSON     = Path("ratio_stats.json")
DATA_CSV       = Path("df.csv")
LEFT_KEY_CANDS = ["ticker","tic","symbol"]
YEAR_CANDS     = ["year","fiscal_year","fyear","report_year"]

# ------------------------------------------------------------------------------
# 3) 加载模型和数据
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model = load_model("bankruptcy_model.h5")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("cat_encoders.pkl")

    with open("preproc_info.json") as f:
        info = json.load(f)

    # 读 df.csv
    data_df = pd.read_csv(DATA_CSV) if DATA_CSV.exists() else None
    ticker_col = None
    if data_df is not None:
        for c in data_df.columns:
            if c.lower() in LEFT_KEY_CANDS:
                ticker_col = c
                break

    # 读文本指标
    extra_df = None
    if EXTRA_INFO_CSV.exists():
        df_tmp = pd.read_csv(EXTRA_INFO_CSV)
        # 若 df_text_final.csv 用 "tic" 而主表是 "ticker" 等
        if ticker_col and "tic" in df_tmp.columns and ticker_col != "tic":
            df_tmp.rename(columns={"tic": ticker_col}, inplace=True)
        extra_df = df_tmp

    # 读行业中位数
    stats = None
    if STATS_JSON.exists():
        with open(STATS_JSON) as f:
            stats = json.load(f)

    return model, scaler, encoders, info, extra_df, stats, data_df, ticker_col

model, scaler, encoders, info, extra_df, ratio_stats, df_raw, TICKER_COL = load_assets()
num_cols, cat_cols, num_means = info["num_cols"], info["cat_cols"], info["num_means"]

# ------------------------------------------------------------------------------
# 4) 工具函数
# ------------------------------------------------------------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).strip())

def strip_unnamed(df: pd.DataFrame):
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

OFFICIAL = {c: normalize(c) for c in [
    "Working Capital (wcap)", "Assets Total(at)", "Current Assets(act)",
    "Current Liabilities - Total (lct)", "Retained Earnings (re)", "ebit",
    "Market Value - Total (mkvalt)", "Liabilities - Total (lt)",
    "Sales/Turnover - Net (sale)", "Operating Activities - Net Cash Flow (oancf)"
]}
M2O = {v: k for k,v in OFFICIAL.items()}

def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: M2O.get(normalize(c), c) for c in df.columns})

def preprocess(df: pd.DataFrame):
    df = strip_unnamed(df).copy()
    for col in num_cols:
        if col not in df.columns:
            df[col] = np.nan

    df[num_cols] = (
        df[num_cols]
        .replace([np.inf,-np.inf], np.nan)
        .fillna(num_means)
        .clip(-1e10,1e10)
        .astype("float32")
    )
    X_num = scaler.transform(df[num_cols])

    X_cat = []
    for c in cat_cols:
        le = encoders[c]
        if c not in df.columns:
            df[c] = "Unknown"
        vals = df[c].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
        X_cat.append(vals.values.astype("int64"))

    return X_cat + [X_num]

# ------------------------------------------------------------------------------
# 5) 计算财务比率
# ------------------------------------------------------------------------------
ratio_names = [
    "liquidity buffer",
    "cumulative profitability",
    "asset profitability",
    "market cushion",
    "asset turnover",
    "cash coverage"
]

def add_ratios(out: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    def col(x):
        return src.get(x, pd.Series(np.nan, index=out.index))

    sdiv = lambda a,b: np.where(b==0, np.nan, a/b)

    wc = col("Working Capital (wcap)")
    if wc is None or wc.isna().all():
        wc = col("Current Assets(act)") - col("Current Liabilities - Total (lct)")

    ratio_df = pd.DataFrame({
        "liquidity buffer":         sdiv(wc,                           col("Assets Total(at)")),
        "cumulative profitability": sdiv(col("Retained Earnings (re)"), col("Assets Total(at)")),
        "asset profitability":      sdiv(col("ebit"),                   col("Assets Total(at)")),
        "market cushion":           sdiv(col("Market Value - Total (mkvalt)"), col("Liabilities - Total (lt)")),
        "asset turnover":           sdiv(col("Sales/Turnover - Net (sale)"),   col("Assets Total(at)")),
        "cash coverage":            sdiv(col("Operating Activities - Net Cash Flow (oancf)"),
                                        col("Liabilities - Total (lt)"))
    }, index=out.index)

    return pd.concat([out.reset_index(drop=True), ratio_df.reset_index(drop=True)], axis=1)

# -- 映射旧字段 X1_WC_TA → 新字段 liquidity buffer
old2new = {
    "X1_WC_TA": "liquidity buffer",
    "X2_RE_TA": "cumulative profitability",
    "X3_EBIT_TA": "asset profitability",
    "X4_MV_TL": "market cushion",
    "X5_SALES_TA": "asset turnover",
    "X6_OCF_TL": "cash coverage"
}

def get_median(rkey, ratio_stats):
    """兼容旧字段名"""
    if ratio_stats and rkey in ratio_stats:
        return ratio_stats[rkey].get("median", None)
    # 否则尝试旧名
    old_keys = [k for k,v in old2new.items() if v==rkey]
    if old_keys:
        old_k = old_keys[0]
        if old_k in ratio_stats:
            return ratio_stats[old_k].get("median", None)
    return None

# ------------------------------------------------------------------------------
# 6) 页面标题 + Model 介绍
# ------------------------------------------------------------------------------
st.title("📉 Bankruptcy Risk Predictor")

st.markdown("**Creators：Yifeng Chen, Syeda Shehrbano Aqeel, Aiden Cliff**")

# Model 1
st.markdown("""
### Model 1 – Textual Topic + Metrics

**Data**: Raw 10-K text + distress label
**Pre-processing**: `CountVectorizer` (stop-words, term-freq filter), plus extra features (word count / complexity / sentiment / similarity)
**Core model**: LDA topics → logistic regression / XGB
**Interpretation**: pyLDAvis for topic themes, etc.
""")

# Model 2
st.markdown("""
### Model 2 – Hybrid Deep Neural Network

**Features**: Cleaned & scaled financial ratios, plus optional categorical embeddings
**Architecture**: Dense → BN → ReLU × n layers → Sigmoid
**Training**: Class weighting for imbalance, EarlyStopping on val AUC
**Output**: Bankruptcy probability (0-1)
""")

# Ratio definitions
st.markdown("""
### Financial Ratio Definitions & Scaling Details

They cover liquidity, cumulative profits, operating profits, market sentiment, sales efficiency and cash flow coverage – giving a balanced snapshot of solvency.

| Ratio                    | Formula                             | Typical range (rule-of-thumb) |
|--------------------------|-------------------------------------|--------------------------------|
| Liquidity buffer         | Working Capital / Total Assets      | 0 – 1                          |
| Cumulative profitability | Retained Earnings / TA             | ≥ 0 preferred                  |
| Asset profitability      | EBIT / TA                           | 0 – 1                          |
| Market cushion           | Market Value / Total Liabilities    | > 1 healthy                    |
| Asset turnover           | Sales / TA                          | 0 – 3                          |
| Cash coverage            | Operating CF / Total Liabilities    | 0 – 1                          |

_All numeric fields are standardized (μ = 0, σ = 1) before feeding the neural net._
""")

# ------------------------------------------------------------------------------
# 7) 方法1：上传 CSV
# ------------------------------------------------------------------------------
st.markdown("#### Method 1: Upload your CSV for batch prediction (optional)")
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    raw_df = pd.read_csv(uploaded)
    raw_df = std_cols(raw_df)
    X = preprocess(raw_df)
    p = model.predict(X).ravel()
    preds = (p>=0.5).astype(int)

    base = pd.DataFrame({
        "Ticker_Uploaded": raw_df[TICKER_COL] if TICKER_COL in raw_df.columns else raw_df.index,
        "Bankruptcy_Probability": np.round(p,4),
        "Prediction": preds
    })
    if extra_df is not None and TICKER_COL in raw_df.columns:
        base = base.merge(extra_df, left_on="Ticker_Uploaded", right_on=TICKER_COL, how="left")

    batch_res = add_ratios(base, raw_df)
    st.subheader("Batch Results Table")
    st.dataframe(batch_res, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=batch_res.to_csv(index=False).encode(),
        file_name="batch_predictions.csv"
    )

st.write("---")

# ------------------------------------------------------------------------------
# 8) 方法2：按 Ticker 搜索
# ------------------------------------------------------------------------------
st.markdown("#### Method 2: Search an existing company by Ticker")

if df_raw is None:
    st.warning("No df.csv found.")
    st.stop()
if not TICKER_COL:
    st.warning("Cannot detect Ticker column in df.csv.")
    st.stop()

col1, col2 = st.columns([3,1])
with col1:
    ticker = st.text_input("Enter Ticker").strip().upper()
with col2:
    clicked = st.button("Search")

if ticker and clicked:
    data = std_cols(df_raw)
    matches = data[data[TICKER_COL].astype(str).str.upper()==ticker]
    if matches.empty:
        st.warning(f"Ticker '{ticker}' not found in df.csv.")
        st.stop()

    # 若有年份
    year_col = None
    for c in data.columns:
        if c.lower() in YEAR_CANDS:
            year_col = c
            break

    if year_col and len(matches[year_col].unique())>1:
        ysel = st.selectbox("Select filing year", sorted(matches[year_col].unique(), reverse=True))
        row = matches[matches[year_col]==ysel].head(1)
    else:
        row = matches.tail(1)

    # 推断
    X = preprocess(row)
    p = model.predict(X).ravel()
    pred = (p>=0.5).astype(int)

    base = pd.DataFrame({
        TICKER_COL: row[TICKER_COL].values,
        "Bankruptcy_Probability": np.round(p,4),
        "Prediction": pred
    })
    full = add_ratios(base, row)

    # 合并文本指标
    if extra_df is not None and TICKER_COL in extra_df.columns:
        full = full.merge(extra_df, on=TICKER_COL, how="left")

    st.subheader("Results Table")
    st.dataframe(full, use_container_width=True)

    # ----- Text-based Metrics -----
    st.markdown("### Text-based Metrics (first row)")
    txt_cols = {
        "distress": "Binary flag from textual analysis (1=high-distress)",
        "word_count": "Total words in MD&A/risk section",
        "word_complexity": "Average syllables per word",
        "sentiment": "Overall Loughran-McDonald sentiment (-1 to 1)",
        "word_similarity": "Cosine sim vs. previous year (lower = bigger change)"
    }
    found_any = False
    if len(full)>0:
        for c in txt_cols:
            if c in full.columns:
                val = full.at[0,c]
                if pd.notna(val):
                    found_any = True
                    st.markdown(f"- **{c}** ({txt_cols[c]}) : `{val}`")
    if not found_any:
        st.info("No text metrics available in this row.")

    # ----- Ratio Interpretation -----
    st.markdown("### Ratio Interpretation (first row)")
    ratio_expl = {
        "liquidity buffer":         "Liquidity > 0 means CA > CL.",
        "cumulative profitability": "Higher = profits financed assets.",
        "asset profitability":      "Efficiency of assets generating EBIT.",
        "market cushion":           "If > 1, market cap exceeds debt.",
        "asset turnover":           "Sales efficiency of asset base.",
        "cash coverage":            "Cash flow ability to service debt."
    }

    for rk in ratio_names:
        if rk not in full.columns:
            continue
        cval = full.at[0, rk]
        if pd.isna(cval):
            st.markdown(f"- **{rk}** : `N/A` — Missing data.")
            continue
        st.markdown(f"- **{rk}** : `{cval:.4g}` — {ratio_expl.get(rk,'')}")

        # 与行业中位数对比 (兼容旧名)
        mval = get_median(rk, ratio_stats)
        if mval is not None and pd.notna(mval):
            chart_df = pd.DataFrame({
                "Type": ["Company","Industry Median"],
                "Value": [cval,mval]
            })
            # ★ 改为水平条形图 + labelOverlap="greedy"，避免文字重叠
            bar = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    y=alt.Y("Type:N", sort=["Company","Industry Median"], title=None),
                    x=alt.X("Value:Q", title=rk, scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelOverlap="greedy")),
                    color=alt.Color("Type:N", legend=None)
                )
                .properties(width=600, height=100)  # 加宽、增高
            )
            st.altair_chart(bar, use_container_width=False)

    
else:
    st.info("Enter ticker and click Search. Probability = chance of bankruptcy within next 3 FYs.")
