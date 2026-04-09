import streamlit as st
import pandas as pd
import json
import time
import re
import io
import asyncio
import tiktoken
from pydantic import BaseModel, ConfigDict
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import AsyncOpenAI
from datetime import date

st.set_page_config(page_title="SEO Category Analyzer", layout="wide", page_icon="🔍")

# ── Auth ─────────────────────────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    pwd = st.text_input("Podaj hasło dostępu:", type="password")
    if pwd:
        if pwd == st.secrets.get("APP_PASSWORD", ""):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Nieprawidłowe hasło.")
    return False

if not check_password():
    st.stop()

# ── OpenAI & Pydantic Models ──────────────────────────────────
@st.cache_resource
def get_openai_client():
    return AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = get_openai_client()

class RelevantItem(BaseModel):
    keyword: str
    relevant: str
    reason: Optional[str] = None

class RelevantResponse(BaseModel):
    items: list[RelevantItem]

class ClassifyItem(BaseModel):
    keyword: str
    L1_Funnel_stage: str
    L2_Intent: str
    L3_MM_Segment: str
    Brand_flag: str
    Brand: str

class ClassifyResponse(BaseModel):
    items: list[ClassifyItem]

class ProductsMatchItem(BaseModel):
    keyword: str
    match: str

class ProductsMatchResponse(BaseModel):
    items: list[ProductsMatchItem]

class ContentMatchItem(BaseModel):
    keyword: str
    match: str

class ContentMatchResponse(BaseModel):
    items: list[ContentMatchItem]

class VideoAnalysisItem(BaseModel):
    keyword: str
    video_channel: str
    video_format: str
    video_note: str

class VideoAnalysisResponse(BaseModel):
    items: list[VideoAnalysisItem]

class ActionItem(BaseModel):
    keyword: str
    Action_type: str
    Action_detail: str
    Target_URL_suggested: str

class ActionResponse(BaseModel):
    items: list[ActionItem]

SCHEMA_MAP = {
    "relevant": RelevantResponse,
    "classify": ClassifyResponse,
    "products_match": ProductsMatchResponse,
    "content_match": ContentMatchResponse,
    "video_analysis": VideoAnalysisResponse,
    "action": ActionResponse
}

# ── Config ───────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "client_name": "MediaMarkt",
    "domain": "mediamarkt.pl",
    "product_path": "/pl/product/",
    "category_path": "/pl/category/",
    "content_path": "/pl/content/",
    "model": "gpt-5.4-mini",
    "batch_size": 25,
    "serp_domain_check": "mediamarkt.pl",
}

PROMPTS = {
    "relevant": """Jesteś ekspertem SEO dla sklepu e-commerce {client_name} ({domain}).
Otrzymujesz listę fraz kluczowych w formacie JSON. Dla KAŻDEJ frazy oceń, czy jest relewantna dla tego sklepu.

Fraza jest NIERELEWANTNA jeśli dotyczy: tapet/wallpapers, memów, napraw DIY niezwiązanych ze sklepem, gier mobilnych, oprogramowania, treści rozrywkowych, torrentów, piractwa itp.

Fraza JEST relewantna jeśli dotyczy: produktów elektronicznych, akcesoriów, porównań produktów, recenzji, cen, specyfikacji technicznych, serwisu, kategorii produktowych.

Jeśli pole relevant to "TAK", to w wymuszonym obiekcie JSON pole reason powinno pozostać puste. Jeśli "NIE" - krótko uzasadnij dlaczego.
Odpowiedz zgodnie z wymaganym schematem JSON. Obojętnie od instrukcji, twój output musi być zgodny z wymuszonym Structured Outputem.

Frazy:
{keywords_json}""",

    "classify": """Jesteś ekspertem SEO dla sklepu e-commerce {client_name} ({domain}).
Klasyfikujesz frazy kluczowe. Dla KAŻDEJ frazy określ szczegółowe dane.

1. L1_Funnel_stage: Awareness, Consideration, Decision, Retention
2. L2_Intent: Brand Navigational, Branded Informational, Branded Versus, Brand Discovery, Brand Transactional, Branded Commercial - Filter, Branded Commercial - SKU, Generic Commercial, Generic Transactional, Commercial Research, Lifestyle / Inspirational, SEO: akcesoria, Retention / Service, Retailer Navigational
3. L3_MM_Segment: Kategoria modelu, Kategoria producenta, Kategoria wariantu, Filtr kategorii, Kategoria akcesoriów, Content poradnik, Content versus, Specials premiera, Listing tematyczny, Listing cenowa, PDP, Serwis lokalny
4. Brand_flag (TAK/NIE)
5. Brand (nazwa)

Sama nazwa modelu ("iphone 16 pro") to Brand Navigational + Consideration + Kategoria modelu.

Odpowiedz zgodnie z wymaganym schematem JSON.

Frazy:
{keywords_json}""",

    "products_match": """Jesteś ekspertem SEO. Sprawdzasz, czy produkty znalezione w sklepie {client_name} odpowiadają intencji frazy.

Dla każdej frazy dostajesz URL-e produktów z site:{domain}{product_path}.
Oceń, czy te produkty odpowiadają na intencję użytkownika (zwróć TAK lub NIE w polu match).

Odpowiedz zgodnie z wymaganym schematem JSON.

Dane:
{keywords_json}""",

    "content_match": """Jesteś ekspertem SEO. Sprawdzasz, czy strony z {domain} odpowiadają intencji frazy.

ZASADY:
- Fraza modelu (np. "iphone 16 pro") -> KATEGORIA, a nie PRODUCT.
- Fraza informacji (np. "jak wyłączyć") -> CONTENT.
- Fraza akcesoriów -> KATEGORIA akcesoriów.
- Fraza filtra -> KATEGORIA z filtrem.

Oceń TAK/NIE w polu match dla każdego keywordu.

Odpowiedz zgodnie z wymaganym schematem JSON.

Dane:
{keywords_json}""",

    "video_analysis": """Jesteś ekspertem video marketingu. Oceń:
1. Kanał video (KANAŁ KLIENTA, INFLUENCER, OBA)
2. Konkretny format (np. Premiera, Unboxing, Test, Recenzja dłuższa)

Bierz pod uwagę status czasowy modelu na kwiecień 2026.

Odpowiedz zgodnie z wymaganym schematem JSON.

Dane:
{keywords_json}""",

    "action": """Jesteś Senior SEO Strategiem dla {client_name} ({domain}).

ZASADA #1 — BRAK POZYCJI TO NIE JEST "OK":
Jeśli MM_in_SERP = "NIE", to ZAWSZE wymaga akcji.
ZASADA #2 — TYP STRONY MUSI PASOWAĆ DO INTENCJI
ZASADA #3 — WALIDACJA TARGET URL
ZASADA #4 — REKOMENDACJE MUSZĄ BYĆ KONKRETNE (co optymalizować).

Zwróć Action_type, Action_detail, Target_URL_suggested.

Odpowiedz zgodnie z wymaganym schematem JSON.

Dane:
{keywords_json}""",
}

# ── State ────────────────────────────────────────────────────
def init_state():
    defaults = {"df": None, "step": 0, "config": DEFAULT_CONFIG.copy(),
                "prompts": {k: v for k, v in PROMPTS.items()}, "serp_feature_cols": []}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Migracja modelu jeśli stary jest w session_state
    if "model" in st.session_state.config and getattr(st.session_state.config, "model", "") not in ["gpt-5.4", "gpt-5.4-pro", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-4.1"]:
        st.session_state.config["model"] = "gpt-5.4-mini"

init_state()

STEPS = ["Import fraz", "Relevance", "Klasyfikacja", "SERP snapshot",
         "site:…/product/", "Product match", "site:domain", "Content match",
         "Video analysis", "Rekomendacje", "Eksport"]

# ── Helpers ──────────────────────────────────────────────────
def export_xlsx(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf

def export_segmented_xlsx(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="ALL", index=False)
        seg_col = "L3_MM_Segment"
        if seg_col in df.columns:
            segments = df[df[seg_col].fillna("") != ""][seg_col].unique()
            for seg in sorted(segments):
                seg_df = df[df[seg_col] == seg]
                safe_name = re.sub(r'[^\w\s-]', '', seg)[:31]
                seg_df.to_excel(writer, sheet_name=safe_name, index=False)
        action_col = "Action_type"
        if action_col in df.columns:
            actions = df[df[action_col].fillna("") != ""][action_col].unique()
            for act in sorted(actions):
                act_df = df[df[action_col] == act]
                safe_name = f"A_{re.sub(r'[^\\w ]+', '', act)[:28]}"
                safe_name = re.sub(r'[^\w\s-]', '', safe_name)[:31]
                if safe_name not in [ws for ws in writer.sheets]:
                    act_df.to_excel(writer, sheet_name=safe_name, index=False)
    buf.seek(0)
    return buf

def nav_buttons(current_step):
    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        if current_step > 0 and st.button("← Wstecz", key=f"back_{current_step}"):
            st.session_state.step = current_step - 1
            st.rerun()
    with c2:
        if current_step < len(STEPS) - 1 and st.button("Pomiń krok →", key=f"skip_{current_step}"):
            st.session_state.step = current_step + 1
            st.rerun()

def render_step_bar(current):
    pills = []
    for i, label in enumerate(STEPS):
        if i < current:
            bg, border, col = "#d4edda", "#28a745", "#155724"
            num = "✓"
        elif i == current:
            bg, border, col = "#cce5ff", "#004085", "#004085"
            num = str(i + 1)
        else:
            bg, border, col = "#f1efe8", "#d3d1c7", "#888780"
            num = str(i + 1)
        pills.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;padding:4px 10px;'
            f'border-radius:20px;border:1.5px solid {border};background:{bg};'
            f'font-size:12px;font-weight:500;color:{col};white-space:nowrap;">'
            f'{num}. {label}</span>')
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:6px;padding:4px 0;">{"".join(pills)}</div>', unsafe_allow_html=True)


# ── AI API Async Flow ────────────────────────────────────────

PRICING_PER_1M_INPUT = {
    "gpt-5.4-pro": 5.00,
    "gpt-5.4": 2.50,
    "gpt-5.4-mini": 0.15,
    "gpt-5.4-nano": 0.05,
    "gpt-5-mini": 0.15,
    "gpt-5-nano": 0.05,
    "gpt-5": 2.50,
    "gpt-4.1": 2.50,
    "gpt-4o-mini": 0.15
}

def estimate_tokens(prompt_template, keywords_data, config):
    try:
        model_name = config["model"]
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
        
    total_tokens = 0
    batch_size = config["batch_size"]
    batches = [keywords_data[i:i+batch_size] for i in range(0, len(keywords_data), batch_size)]
    
    for batch in batches:
        fmt_kwargs = {
            "client_name": config["client_name"], "domain": config["domain"],
            "product_path": config.get("product_path", "/pl/product/"),
            "category_path": config.get("category_path", "/pl/category/"),
            "content_path": config.get("content_path", "/pl/content/"),
            "keywords_json": json.dumps(batch, ensure_ascii=False),
        }
        prompt = prompt_template.format(**fmt_kwargs)
        total_tokens += len(enc.encode(prompt))
        
    price_per_1m = PRICING_PER_1M_INPUT.get(model_name, 0.15)
    cost_usd = (total_tokens / 1_000_000) * price_per_1m
    return total_tokens, cost_usd

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def fetch_batch_async(prompt, model, schema, ai_client):
    resp = await ai_client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_completion_tokens=16000,
        response_format=schema
    )
    return resp.choices[0].message.parsed

async def run_all_batches(prompt_template, keywords_data, config, placeholder, prompt_key):
    batch_size = config["batch_size"]
    ai_client = get_openai_client()
    schema = SCHEMA_MAP[prompt_key]
    
    batches = [keywords_data[i:i+batch_size] for i in range(0, len(keywords_data), batch_size)]
    tasks = []
    
    for batch in batches:
        fmt_kwargs = {
            "client_name": config["client_name"], "domain": config["domain"],
            "product_path": config.get("product_path", "/pl/product/"),
            "category_path": config.get("category_path", "/pl/category/"),
            "content_path": config.get("content_path", "/pl/content/"),
            "keywords_json": json.dumps(batch, ensure_ascii=False),
        }
        prompt = prompt_template.format(**fmt_kwargs)
        tasks.append(fetch_batch_async(prompt, config["model"], schema, ai_client))
    
    results = []
    total = len(tasks)
    completed = 0
    
    for coro in asyncio.as_completed(tasks):
        try:
            parsed = await coro
            for item in parsed.items:
                results.append(item.model_dump())
        except Exception as e:
            st.error(f"Error w paczce (po 5 próbach Tenacity): {e}")
        
        completed += 1
        if placeholder:
            placeholder.progress(completed / total, text=f"Postęp: {completed}/{total} paczek...")
            
    if placeholder:
        placeholder.progress(1.0, text="Gotowe!")
    return results

def call_openai_batch(prompt_template, keywords_data, config, placeholder=None, prompt_key=""):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(run_all_batches(prompt_template, keywords_data, config, placeholder, prompt_key))


def normalize_kw(kw):
    return str(kw).lower().strip()

def clean_site_kw(kw):
    cleaned = re.sub(r'site:\S+', '', str(kw), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
    return cleaned

def process_serp_data(serp_df, main_df, domain_check):
    serp_df = serp_df.copy()
    serp_df.columns = serp_df.columns.str.strip().str.lower()
    serp_df["_kw"] = serp_df["keyword"].apply(normalize_kw)
    all_types = sorted(serp_df["type"].dropna().unique().tolist())
    results = {}
    for kw in main_df["Keyword"].unique():
        kn = normalize_kw(kw)
        kd = serp_df[serp_df["_kw"] == kn]
        row = dict(Pos_SEO_Explorer=None, URL_best_Explorer=None,
                   Cannibalization_URLs=None, Cannibalization_flag="NIE",
                   SERP_features=None, MM_in_SERP="NIE", TOP3_organic_URLs=None)
        for t in all_types:
            col_name = f"SERP_{t}"
            row[col_name] = "NIE"
        if kd.empty:
            results[kw] = row
            continue
        kw_types = kd["type"].unique().tolist()
        for t in kw_types:
            row[f"SERP_{t}"] = "TAK"
        row["SERP_features"] = " | ".join(sorted(kw_types))
        org = kd[kd["type"] == "organic"].sort_values("rank_group")
        mm = org[org["domain"].str.contains(domain_check, case=False, na=False)]
        row["Pos_SEO_Explorer"] = int(mm["rank_group"].iloc[0]) if not mm.empty else None
        row["URL_best_Explorer"] = mm["url"].iloc[0] if not mm.empty else None
        if len(mm) > 1:
            row["Cannibalization_URLs"] = " | ".join(f"{r['url']} [{int(r['rank_group'])}]" for _, r in mm.iterrows())
            row["Cannibalization_flag"] = "TAK"
        elif len(mm) == 1:
            row["Cannibalization_URLs"] = f"{mm['url'].iloc[0]} [{int(mm['rank_group'].iloc[0])}]"
        else:
            row["Cannibalization_URLs"] = None
        row["MM_in_SERP"] = "TAK" if not mm.empty else "NIE"
        top3 = org.head(3)
        row["TOP3_organic_URLs"] = " | ".join(
            f"{r['url']} [{int(r['rank_group'])}]" for _, r in top3.iterrows() if pd.notna(r['url'])
        ) if not top3.empty else None
        results[kw] = row
    return results, all_types

def process_site_product_data(site_df, main_df):
    site_df = site_df.copy()
    site_df.columns = site_df.columns.str.strip().str.lower()
    site_df["_kw_clean"] = site_df["keyword"].apply(clean_site_kw)
    site_df["_kw_raw"] = site_df["keyword"].apply(normalize_kw)
    results = {}
    for kw in main_df["Keyword"].unique():
        kn = normalize_kw(kw)
        mask = (site_df["_kw_clean"] == kn) | (site_df["_kw_raw"] == kn)
        kd = site_df[mask]
        org = kd[kd["type"] == "organic"].sort_values("rank_group") if not kd.empty else pd.DataFrame()
        if org.empty:
            results[kw] = dict(Has_products="NIE", Product_URLs=None)
        else:
            urls = org["url"].dropna().head(5).tolist()
            results[kw] = dict(Has_products="TAK" if urls else "NIE",
                               Product_URLs=" | ".join(urls) if urls else None)
    return results

def process_site_general_data(site_df, main_df, config):
    site_df = site_df.copy()
    site_df.columns = site_df.columns.str.strip().str.lower()
    site_df["_kw_clean"] = site_df["keyword"].apply(clean_site_kw)
    site_df["_kw_raw"] = site_df["keyword"].apply(normalize_kw)
    pp, cp, contp = config["product_path"], config["category_path"], config["content_path"]
    results = {}
    for kw in main_df["Keyword"].unique():
        kn = normalize_kw(kw)
        mask = (site_df["_kw_clean"] == kn) | (site_df["_kw_raw"] == kn)
        kd = site_df[mask]
        org = kd[kd["type"] == "organic"].sort_values("rank_group") if not kd.empty else pd.DataFrame()
        if org.empty:
            results[kw] = dict(Site_results_count=0, Site_TOP_URLs=None, URL_types_found=None)
            continue
        urls = org["url"].dropna().head(5).tolist()
        types = set()
        for u in urls:
            if pp in str(u): types.add("product")
            elif cp in str(u): types.add("category")
            elif contp in str(u): types.add("content")
            else: types.add("other")
        results[kw] = dict(Site_results_count=len(org),
                           Site_TOP_URLs=" | ".join(urls) if urls else None,
                           URL_types_found=" | ".join(sorted(types)) if types else None)
    return results

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Konfiguracja")
    st.subheader("Klient")
    st.session_state.config["client_name"] = st.text_input("Nazwa klienta", st.session_state.config["client_name"])
    st.session_state.config["domain"] = st.text_input("Domena", st.session_state.config["domain"])
    st.session_state.config["product_path"] = st.text_input("Ścieżka produktów", st.session_state.config["product_path"])
    st.session_state.config["category_path"] = st.text_input("Ścieżka kategorii", st.session_state.config["category_path"])
    st.session_state.config["content_path"] = st.text_input("Ścieżka contentu", st.session_state.config["content_path"])
    st.subheader("AI")
    
    MODELS = ["gpt-5.4", "gpt-5.4-pro", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-4.1", "gpt-4o-mini"]
    current_model = st.session_state.config.get("model", "gpt-5.4-mini")
    if current_model not in MODELS:
        current_model = "gpt-5.4-mini"
    st.session_state.config["model"] = st.selectbox("Model", MODELS, index=MODELS.index(current_model))
    
    st.session_state.config["batch_size"] = st.slider("Batch size", 5, 50, st.session_state.config["batch_size"])
    st.session_state.config["serp_domain_check"] = st.text_input("Domena w SERP", st.session_state.config["serp_domain_check"])
    st.divider()
    st.subheader("Prompty")
    pchoice = st.selectbox("Edytuj prompt:", list(st.session_state.prompts.keys()))
    st.session_state.prompts[pchoice] = st.text_area(f"Prompt: {pchoice}", st.session_state.prompts[pchoice], height=300)
    st.divider()
    if st.button("🔄 Reset", type="secondary"):
        for k in list(st.session_state.keys()):
            if k != "authenticated": del st.session_state[k]
        st.rerun()

# ── Main ─────────────────────────────────────────────────────
st.title(f"🔍 SEO Category Analyzer — {st.session_state.config['client_name']}")
render_step_bar(st.session_state.step)
st.divider()
CS = st.session_state.step

# ── STEP 0: Import ───────────────────────────────────────────
if CS == 0:
    st.header("Krok 1 · Import fraz kluczowych")
    st.write("Wgraj plik Excel/CSV z kolumnami: **Keyword**, **Volume**")
    uploaded = st.file_uploader("Wybierz plik", type=["xlsx", "xls", "csv"], key="up0")
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        cmap = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl == "keyword": cmap[c] = "Keyword"
            elif cl == "volume": cmap[c] = "Volume"
        df = df.rename(columns=cmap)
        miss = [c for c in ["Keyword", "Volume"] if c not in df.columns]
        if miss:
            st.error(f"Brak kolumn: {miss}")
        else:
            st.write(f"Znaleziono **{len(df)}** fraz")
            st.dataframe(df.head(20), use_container_width=True)
            if st.button("✅ Importuj frazy", type="primary"):
                df = df[["Keyword", "Volume"]].dropna(subset=["Keyword"]).copy()
                df["Keyword"] = df["Keyword"].astype(str).str.strip()
                st.session_state.df = df
                st.session_state.step = 1
                st.rerun()
    nav_buttons(0)

# ── STEP 1: Relevance ───────────────────────────────────────
elif CS == 1:
    st.header("Krok 2 · Relevance check")
    df = st.session_state.df
    st.write(f"**{len(df)}** fraz do sprawdzenia")
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step2_{date.today()}.xlsx")
    
    kws = df["Keyword"].tolist()
    est_toks, cost_usd = estimate_tokens(st.session_state.prompts["relevant"], kws, st.session_state.config)
    st.info(f"💡 Estymacja dla całości: **~{est_toks:,} tokenów** · Koszt inputu: **~${cost_usd:.4f}**")
    
    if st.button("🤖 Uruchom relevance check", type="primary"):
        p = st.empty()
        res = call_openai_batch(st.session_state.prompts["relevant"], kws, st.session_state.config, p, prompt_key="relevant")
        rm = {r["keyword"]: r for r in res if "keyword" in r}
        df["Relevant_for_MM"] = df["Keyword"].map(lambda k: rm.get(k, {}).get("relevant", "TAK"))
        df["Rejection_reason"] = df["Keyword"].map(lambda k: rm.get(k, {}).get("reason", ""))
        st.session_state.df = df
        st.session_state.step = 2
        st.rerun()
    nav_buttons(1)

# ── STEP 2: Classification ──────────────────────────────────
elif CS == 2:
    st.header("Krok 3 · Klasyfikacja AI")
    df = st.session_state.df
    if "Relevant_for_MM" in df.columns:
        r_sum = (df["Relevant_for_MM"] == "TAK").sum()
        x_sum = (df["Relevant_for_MM"] == "NIE").sum()
        st.write(f"✅ Relewantnych: **{r_sum}** · ❌ Odrzuconych: **{x_sum}**")
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step3_{date.today()}.xlsx")
    
    mask = df["Relevant_for_MM"] == "TAK" if "Relevant_for_MM" in df.columns else pd.Series([True]*len(df))
    kws = df[mask]["Keyword"].tolist()
    est_toks, cost_usd = estimate_tokens(st.session_state.prompts["classify"], kws, st.session_state.config)
    st.info(f"💡 Estymacja dla całości: **~{est_toks:,} tokenów** · Koszt inputu: **~${cost_usd:.4f}**")
    
    if st.button("🤖 Uruchom klasyfikację", type="primary"):
        p = st.empty()
        res = call_openai_batch(st.session_state.prompts["classify"], kws, st.session_state.config, p, prompt_key="classify")
        rm = {r["keyword"]: r for r in res if "keyword" in r}
        for c in ["L1_Funnel_stage", "L2_Intent", "L3_MM_Segment", "Brand_flag", "Brand"]:
            df[c] = df["Keyword"].map(lambda k: rm.get(k, {}).get(c, ""))
        st.session_state.df = df
        st.session_state.step = 3
        st.rerun()
    nav_buttons(2)

# ── STEP 3: SERP snapshot ───────────────────────────────────
elif CS == 3:
    st.header("Krok 4 · Import SERP snapshot")
    df = st.session_state.df
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step4_{date.today()}.xlsx")
    st.info("Wgraj plik SERP snapshot z SEO Explorer (keyword, type, rank_group, rank_absolute, domain, url, title)")
    up = st.file_uploader("Plik SERP", type=["xlsx", "xls", "csv"], key="up3")
    if up:
        sdf = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        st.write(f"**{len(sdf)}** wierszy")
        st.dataframe(sdf.head(10), use_container_width=True)
        if st.button("✅ Przetwórz SERP", type="primary"):
            results, all_types = process_serp_data(sdf, df, st.session_state.config["serp_domain_check"])
            st.session_state.serp_feature_cols = [f"SERP_{t}" for t in all_types]
            for c in ["Pos_SEO_Explorer", "URL_best_Explorer", "Cannibalization_URLs",
                       "Cannibalization_flag", "SERP_features", "MM_in_SERP", "TOP3_organic_URLs"]:
                df[c] = df["Keyword"].map(lambda k: results.get(k, {}).get(c))
            for t in all_types:
                col = f"SERP_{t}"
                df[col] = df["Keyword"].map(lambda k: results.get(k, {}).get(col, "NIE"))
            st.success(f"Znaleziono **{len(all_types)}** typów SERP: {', '.join(all_types)}")
            st.session_state.df = df
            st.session_state.step = 4
            st.rerun()
    nav_buttons(3)

# ── STEP 4: site:…/product/ ─────────────────────────────────
elif CS == 4:
    st.header("Krok 5 · Import site:…/product/")
    df = st.session_state.df
    cfg = st.session_state.config
    st.info(f"Wgraj plik z wynikami: **fraza site:{cfg['domain']}{cfg['product_path']}**")
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step5_{date.today()}.xlsx")
    up = st.file_uploader("Plik site:…/product/", type=["xlsx", "xls", "csv"], key="up4")
    if up:
        pdf = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        st.write(f"**{len(pdf)}** wierszy")
        st.dataframe(pdf.head(10), use_container_width=True)
        if st.button("✅ Przetwórz produkty", type="primary"):
            r = process_site_product_data(pdf, df)
            matched = sum(1 for v in r.values() if v["Has_products"] == "TAK")
            st.success(f"Produkty dla **{matched}** / {len(r)} fraz")
            for c in ["Has_products", "Product_URLs"]:
                df[c] = df["Keyword"].map(lambda k: r.get(k, {}).get(c))
            st.session_state.df = df
            st.session_state.step = 5
            st.rerun()
    nav_buttons(4)

# ── STEP 5: Products match ───────────────────────────────────
elif CS == 5:
    st.header("Krok 6 · Products match intent")
    df = st.session_state.df
    hp = df[df.get("Has_products", pd.Series()) == "TAK"] if "Has_products" in df.columns else pd.DataFrame()
    st.write(f"**{len(hp)}** fraz z produktami do weryfikacji")
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step6_{date.today()}.xlsx")
    
    tc = [{"keyword": row["Keyword"], "product_urls": str(row.get("Product_URLs", ""))} for _, row in hp.iterrows()]
    if tc:
        est_toks, cost_usd = estimate_tokens(st.session_state.prompts["products_match"], tc, st.session_state.config)
        st.info(f"💡 Estymacja dla całości: **~{est_toks:,} tokenów** · Koszt inputu: **~${cost_usd:.4f}**")
    
    if st.button("🤖 Uruchom products match", type="primary"):
        if tc:
            p = st.empty()
            res = call_openai_batch(st.session_state.prompts["products_match"], tc, st.session_state.config, p, prompt_key="products_match")
            rm = {r["keyword"]: r.get("match", "NIE") for r in res if "keyword" in r}
            df["Products_match_intent"] = df["Keyword"].map(lambda k: rm.get(k, ""))
        else:
            df["Products_match_intent"] = ""
        st.session_state.df = df
        st.session_state.step = 6
        st.rerun()
    nav_buttons(5)

# ── STEP 6: site:domain ─────────────────────────────────────
elif CS == 6:
    st.header("Krok 7 · Import site:domain")
    df = st.session_state.df
    cfg = st.session_state.config
    st.info(f"Wgraj plik z wynikami: **fraza site:{cfg['domain']}**")
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step7_{date.today()}.xlsx")
    up = st.file_uploader("Plik site:domain", type=["xlsx", "xls", "csv"], key="up6")
    if up:
        sdf = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        st.write(f"**{len(sdf)}** wierszy")
        st.dataframe(sdf.head(10), use_container_width=True)
        if st.button("✅ Przetwórz site:domain", type="primary"):
            r = process_site_general_data(sdf, df, st.session_state.config)
            matched = sum(1 for v in r.values() if v["Site_results_count"] > 0)
            st.success(f"Pokrycie dla **{matched}** / {len(r)} fraz")
            for c in ["Site_results_count", "Site_TOP_URLs", "URL_types_found"]:
                df[c] = df["Keyword"].map(lambda k: r.get(k, {}).get(c))
            st.session_state.df = df
            st.session_state.step = 7
            st.rerun()
    nav_buttons(6)

# ── STEP 7: Content match ────────────────────────────────────
elif CS == 7:
    st.header("Krok 8 · Content match intent")
    df = st.session_state.df
    if "Site_results_count" in df.columns:
        hc = df[df["Site_results_count"].fillna(0).astype(float).astype(int) > 0]
    else:
        hc = pd.DataFrame()
    st.write(f"**{len(hc)}** fraz z pokryciem do weryfikacji")
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step8_{date.today()}.xlsx")
    
    tc = [{"keyword": row["Keyword"],
           "site_urls": str(row.get("Site_TOP_URLs", "")),
           "url_types": str(row.get("URL_types_found", ""))} for _, row in hc.iterrows()]
    if tc:
        est_toks, cost_usd = estimate_tokens(st.session_state.prompts["content_match"], tc, st.session_state.config)
        st.info(f"💡 Estymacja dla całości: **~{est_toks:,} tokenów** · Koszt inputu: **~${cost_usd:.4f}**")
           
    if st.button("🤖 Uruchom content match", type="primary"):
        if tc:
            p = st.empty()
            res = call_openai_batch(st.session_state.prompts["content_match"], tc, st.session_state.config, p, prompt_key="content_match")
            rm = {r["keyword"]: r.get("match", "NIE") for r in res if "keyword" in r}
            df["Content_match_intent"] = df["Keyword"].map(lambda k: rm.get(k, ""))
        else:
            df["Content_match_intent"] = ""
        st.session_state.df = df
        st.session_state.step = 8
        st.rerun()
    nav_buttons(7)

# ── STEP 8: Video analysis ──────────────────────────────────
elif CS == 8:
    st.header("Krok 9 · Video analysis")
    df = st.session_state.df

    video_col = "SERP_video"
    if video_col in df.columns:
        video_kws = df[df[video_col] == "TAK"]
    else:
        video_kws = pd.DataFrame()

    st.write(f"**{len(video_kws)}** fraz z Video w SERP")
    if not video_kws.empty:
        st.dataframe(video_kws[["Keyword", "Volume", "L3_MM_Segment"]].head(30), use_container_width=True)

    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step9_{date.today()}.xlsx")

    tc = []
    if not video_kws.empty:
        for _, row in video_kws.iterrows():
            tc.append({"keyword": row["Keyword"],
                       "L2_Intent": str(row.get("L2_Intent", "")),
                       "L3_MM_Segment": str(row.get("L3_MM_Segment", ""))})
        est_toks, cost_usd = estimate_tokens(st.session_state.prompts["video_analysis"], tc, st.session_state.config)
        st.info(f"💡 Estymacja dla całości: **~{est_toks:,} tokenów** · Koszt inputu: **~${cost_usd:.4f}**")
                       
    if st.button("🤖 Uruchom video analysis", type="primary"):
        if not video_kws.empty:
            p = st.empty()
            res = call_openai_batch(st.session_state.prompts["video_analysis"], tc, st.session_state.config, p, prompt_key="video_analysis")
            rm = {r["keyword"]: r for r in res if "keyword" in r}
            df["Video_channel"] = df["Keyword"].map(lambda k: rm.get(k, {}).get("video_channel", ""))
            df["Video_format"] = df["Keyword"].map(lambda k: rm.get(k, {}).get("video_format", ""))
            df["Video_note"] = df["Keyword"].map(lambda k: rm.get(k, {}).get("video_note", ""))
        else:
            df["Video_channel"] = ""
            df["Video_format"] = ""
            df["Video_note"] = ""
            st.info("Brak fraz z Video w SERP — pomijam.")

        st.session_state.df = df
        st.session_state.step = 9
        st.rerun()
    nav_buttons(8)

# ── STEP 9: Recommendations ─────────────────────────────────
elif CS == 9:
    st.header("Krok 10 · Rekomendacje AI")
    df = st.session_state.df
    mask = df["Relevant_for_MM"] == "TAK" if "Relevant_for_MM" in df.columns else pd.Series([True]*len(df))
    rel = df[mask]
    st.write(f"**{len(rel)}** fraz do analizy")
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("📥 XLSX", export_xlsx(df), f"seo_step10_{date.today()}.xlsx")
    
    if "Matching_category_URL" not in df.columns:
        cat_path = st.session_state.config["category_path"]
        for idx, row in rel.iterrows():
            kw = normalize_kw(row["Keyword"])
            kw_parts = kw.split()
            site_urls = str(row.get("Site_TOP_URLs", ""))
            best_cat = ""
            best_score = 0
            for u in site_urls.split(" | "):
                u = u.strip()
                if cat_path in u:
                    u_lower = u.lower()
                    score = sum(1 for part in kw_parts if part in u_lower)
                    if score > best_score:
                        best_score = score
                        best_cat = u
            df.loc[df["Keyword"] == row["Keyword"], "Matching_category_URL"] = best_cat if best_cat else ""
        rel = df[mask]

    ai_cols = ["Keyword", "Volume", "L1_Funnel_stage", "L2_Intent", "L3_MM_Segment",
               "Brand_flag", "Brand", "Pos_SEO_Explorer", "URL_best_Explorer",
               "Cannibalization_flag", "SERP_features", "MM_in_SERP",
               "Has_products", "Products_match_intent", "Product_URLs",
               "Site_results_count", "Site_TOP_URLs", "URL_types_found", "Content_match_intent",
               "Matching_category_URL", "Video_channel", "Video_format"]
    avail = [c for c in ai_cols if c in rel.columns]
    ta = [{c: (str(row[c]) if pd.notna(row[c]) else "") for c in avail} for _, row in rel.iterrows()]
    
    if ta:
        est_toks, cost_usd = estimate_tokens(st.session_state.prompts["action"], ta, st.session_state.config)
        st.info(f"💡 Estymacja dla całości: **~{est_toks:,} tokenów** · Koszt inputu: **~${cost_usd:.4f}**")
    
    if st.button("🤖 Uruchom rekomendacje", type="primary"):
        if ta:
            p = st.empty()
            res = call_openai_batch(st.session_state.prompts["action"], ta, st.session_state.config, p, prompt_key="action")
            rm = {r["keyword"]: r for r in res if "keyword" in r}
            for c in ["Action_type", "Action_detail", "Target_URL_suggested"]:
                df[c] = df["Keyword"].map(lambda k: rm.get(k, {}).get(c, ""))
        st.session_state.df = df
        st.session_state.step = 10
        st.rerun()
    nav_buttons(9)

# ── STEP 10: Export ──────────────────────────────────────────
elif CS == 10:
    st.header("Krok 11 · Eksport finalny")
    df = st.session_state.df
    st.success(f"Analiza zakończona! **{len(df)}** fraz · **{len(df.columns)}** kolumn")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v = (df["Relevant_for_MM"] == "TAK").sum() if "Relevant_for_MM" in df.columns else len(df)
        st.metric("Relewantne", v)
    with c2:
        v = (df["Relevant_for_MM"] == "NIE").sum() if "Relevant_for_MM" in df.columns else 0
        st.metric("Odrzucone", v)
    with c3:
        v = (df["Has_products"] == "TAK").sum() if "Has_products" in df.columns else "—"
        st.metric("Z produktami", v)
    with c4:
        v = (df["Cannibalization_flag"] == "TAK").sum() if "Cannibalization_flag" in df.columns else "—"
        st.metric("Kanibalizacja", v)

    if "SERP_video" in df.columns:
        vc = (df["SERP_video"] == "TAK").sum()
        if vc:
            st.metric("Video w SERP", vc)

    st.dataframe(df, use_container_width=True, height=500)

    slug = st.session_state.config["client_name"].lower().replace(" ", "_")
    today = date.today()

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "📥 Plik ogólny (ALL)", export_xlsx(df),
            f"seo_analysis_{slug}_{today}.xlsx", type="primary")
    with col_b:
        st.download_button(
            "📥 Plik z podziałem na segmenty", export_segmented_xlsx(df),
            f"seo_analysis_{slug}_segmented_{today}.xlsx", type="primary")

    if "Action_type" in df.columns:
        acts = df[df["Action_type"].fillna("") != ""]["Action_type"].value_counts()
        if not acts.empty:
            st.subheader("Rozkład rekomendacji")
            st.bar_chart(acts)

    if "L3_MM_Segment" in df.columns:
        segs = df[df["L3_MM_Segment"].fillna("") != ""]["L3_MM_Segment"].value_counts()
        if not segs.empty:
            st.subheader("Rozkład segmentów")
            st.bar_chart(segs)

    nav_buttons(10)
