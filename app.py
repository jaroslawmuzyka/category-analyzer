import streamlit as st
import pandas as pd
import json
import time
import io
from openai import OpenAI
from datetime import date

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="SEO Category Analyzer", layout="wide", page_icon="🔍")

# ── Auth gate ────────────────────────────────────────────────
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

# ── OpenAI client ────────────────────────────────────────────
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = get_openai_client()

# ── Defaults ─────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "client_name": "MediaMarkt",
    "domain": "mediamarkt.pl",
    "product_path": "/pl/product/",
    "category_path": "/pl/category/",
    "content_path": "/pl/content/",
    "model": "gpt-4o-mini",
    "batch_size": 25,
    "serp_domain_check": "mediamarkt.pl",
}

PROMPTS = {
    "relevant": """Jesteś ekspertem SEO dla sklepu e-commerce {client_name} ({domain}).
Otrzymujesz listę fraz kluczowych w formacie JSON. Dla KAŻDEJ frazy oceń, czy jest relewantna dla tego sklepu.

Fraza jest NIERELEWANTNA jeśli dotyczy: tapet/wallpapers, memów, napraw DIY niezwiązanych ze sklepem, gier mobilnych, oprogramowania, treści rozrywkowych, torrentów, piractwa itp.

Fraza JEST relewantna jeśli dotyczy: produktów elektronicznych, akcesoriów, porównań produktów, recenzji, cen, specyfikacji technicznych, serwisu, kategorii produktowych.

Odpowiedz WYŁĄCZNIE jako JSON array. Dla każdej frazy zwróć obiekt:
{{"keyword": "fraza", "relevant": "TAK" lub "NIE", "reason": "krótkie uzasadnienie jeśli NIE, puste jeśli TAK"}}

Frazy do oceny:
{keywords_json}""",

    "classify": """Jesteś ekspertem SEO dla sklepu e-commerce {client_name} ({domain}).
Klasyfikujesz frazy kluczowe. Dla KAŻDEJ frazy określ:

1. L1_Funnel_stage — etap lejka:
   - Awareness (użytkownik dopiero szuka informacji, premiery, nowości)
   - Consideration (porównuje, szuka recenzji, modeli)
   - Decision (chce kupić, szuka ceny, konkretnego SKU, filtruje)
   - Retention (serwis, wymiana baterii, naprawy, wsparcie)

2. L2_Intent — intencja:
   - Brand Navigational (szuka konkretnego modelu/marki)
   - Branded Informational (szuka informacji o konkretnym produkcie marki)
   - Branded Versus (porównanie modeli branded)
   - Brand Discovery (szuka marki ogólnie: "iphone", "samsung galaxy")
   - Brand Transactional (chce kupić konkretny wariant: "iphone 15 pro 256gb cena")
   - Branded Commercial - Filter (filtr: kolor, pamięć, wariant)
   - Branded Commercial - SKU (konkretny SKU/EAN)
   - Generic Commercial (szuka kategorii bez brandu: "smartfon do 2000 zł")
   - Generic Transactional (chce kupić generycznie: "kup smartfon", "tani telefon")
   - Commercial Research (porównania, rankingi: "najlepszy smartfon 2025")
   - Lifestyle / Inspirational (premiery, trendy, nowości)
   - SEO: akcesoria (etui, ładowarki, folie, kable do konkretnych modeli)
   - Retention / Service (serwis, naprawa, wymiana baterii, gwarancja)
   - Retailer Navigational (szuka konkretnego sklepu: "mediamarkt smartfony")

3. L3_MM_Segment — segment w sklepie:
   - Kategoria modelu (strona konkretnego modelu w sklepie)
   - Kategoria producenta (strona marki: "Smartfony Apple")
   - Kategoria wariantu (wariant z ceną: "iPhone 15 Pro 256GB cena")
   - Filtr kategorii (filtrowanie: kolor, pamięć, cena, system)
   - Kategoria akcesoriów (etui, folie, ładowarki)
   - Content poradnik (blog: "jak wyłączyć iPhone", "jak zresetować")
   - Content versus (porównania: "iPhone 15 vs 16")
   - Specials premiera (premiery, zapowiedzi nowych modeli)
   - Listing tematyczny (ranking, zestawienie: "najlepsze smartfony 2025")
   - Listing cenowa (cenówki: "smartfon do 1000 zł")
   - PDP (strona konkretnego produktu SKU)
   - Serwis lokalny (naprawa, wymiana baterii, serwis)

4. Brand_flag — TAK/NIE czy fraza zawiera nazwę marki
5. Brand — nazwa marki (Apple, Samsung, Xiaomi, POCO, Realme, OnePlus, Google, Nothing, Motorola, Nokia, Honor, Sony, OPPO, Vivo, Hammer, Huawei) lub puste

Odpowiedz WYŁĄCZNIE jako JSON array:
{{"keyword": "fraza", "L1_Funnel_stage": "...", "L2_Intent": "...", "L3_MM_Segment": "...", "Brand_flag": "TAK/NIE", "Brand": "..."}}

Frazy:
{keywords_json}""",

    "products_match": """Jesteś ekspertem SEO. Sprawdzasz, czy produkty znalezione w sklepie {client_name} odpowiadają intencji frazy.

Dla każdej frazy dostajesz listę URL-i produktów. Oceń, czy te produkty faktycznie odpowiadają na to, czego szuka użytkownik.

Przykłady:
- Fraza "etui iphone 16 pro" → URL z etui na iPhone 16 Pro → TAK
- Fraza "iphone 16 pro" → URL z iPhone 16 Pro → TAK
- Fraza "iphone 16 pro" → URL z etui/folią → NIE (użytkownik szuka telefonu, nie akcesoriów)
- Fraza "ładowarka do samsung" → URL z ładowarką Samsung → TAK

Odpowiedz WYŁĄCZNIE jako JSON array:
{{"keyword": "fraza", "match": "TAK" lub "NIE"}}

Dane:
{keywords_json}""",

    "content_match": """Jesteś ekspertem SEO. Sprawdzasz, czy strony znalezione w domenie {domain} odpowiadają intencji frazy.

Analizujesz URL-e (ich ścieżkę i tytuły) i oceniasz czy pasują do intencji użytkownika.

Przykłady:
- Fraza "smartfony do 2000 zł" → URL /pl/category/smartfony z filtrami cenowymi → TAK
- Fraza "jak zresetować iphone" → URL /pl/content/iphone-jak-zresetowac → TAK
- Fraza "samsung galaxy a56" → URL /pl/category/smartfony (ogólna) → NIE (brak dedykowanej strony modelu)
- Fraza "etui iphone 16" → URL /pl/product/smartfon-iphone-16 → NIE (to telefon, nie etui)

Odpowiedz WYŁĄCZNIE jako JSON array:
{{"keyword": "fraza", "match": "TAK" lub "NIE"}}

Dane:
{keywords_json}""",

    "action": """Jesteś Senior SEO Strategiem dla sklepu {client_name} ({domain}).
Na podstawie WSZYSTKICH zebranych danych o frazie, określ rekomendowaną akcję.

Kontekst kolumn:
- Has_products: czy sklep ma produkty na tę frazę
- Products_match_intent: czy te produkty pasują do intencji
- URL_types_found: jakie typy stron sklep już ma (product/category/content)
- Content_match_intent: czy istniejące strony odpowiadają na intencję
- L3_MM_Segment: do jakiego segmentu należy fraza
- Cannibalization_flag: czy jest kanibalizacja

Możliwe Action_type:
- Nowa podkategoria — brak strony kategorii, ale są produkty
- Nowy filtr — potrzebny filtr na istniejącej kategorii (kolor, pamięć, cena)
- Nowy wpis blogowy — fraza informacyjna/poradnikowa bez pokrycia
- Optymalizacja istniejącej — jest strona, ale nie odpowiada na intencję lub słabo rankuje
- Nowy landing — potrzebna dedykowana strona (premiera, versus, ranking)
- Redirect — kanibalizacja, trzeba przekierować
- Brak akcji — wszystko OK lub fraza nieistotna

Odpowiedz WYŁĄCZNIE jako JSON array:
{{"keyword": "fraza", "Action_type": "...", "Action_detail": "szczegółowy opis co zrobić", "Target_URL_suggested": "proponowany URL"}}

Dane:
{keywords_json}""",
}


# ── Session state init ───────────────────────────────────────
def init_state():
    defaults = {
        "df": None,
        "step": 0,
        "config": DEFAULT_CONFIG.copy(),
        "prompts": {k: v for k, v in PROMPTS.items()},
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

STEPS = [
    "1. Import fraz",
    "2. Relevance check",
    "3. Klasyfikacja AI",
    "4. Import SERP (SEO Explorer)",
    "5. Import site:…/product/",
    "6. Products match intent",
    "7. Import site:domain",
    "8. Content match intent",
    "9. Rekomendacje AI",
    "10. Eksport finalny",
]


# ── Helpers ──────────────────────────────────────────────────
def export_xlsx(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf

def call_openai_batch(prompt_template, keywords_data, config, placeholder=None):
    batch_size = config["batch_size"]
    results = []
    total = len(keywords_data)
    batches = [keywords_data[i:i+batch_size] for i in range(0, total, batch_size)]

    for idx, batch in enumerate(batches):
        if placeholder:
            placeholder.progress((idx) / len(batches), text=f"Batch {idx+1}/{len(batches)} ({len(batch)} fraz)...")

        prompt = prompt_template.format(
            client_name=config["client_name"],
            domain=config["domain"],
            keywords_json=json.dumps(batch, ensure_ascii=False),
        )

        try:
            resp = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=16000,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
            parsed = json.loads(raw)
            results.extend(parsed)
        except json.JSONDecodeError as e:
            st.warning(f"Batch {idx+1}: błąd parsowania JSON — {e}. Próbuję ponownie...")
            try:
                resp2 = client.chat.completions.create(
                    model=config["model"],
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": "Twoja odpowiedź nie była poprawnym JSONem. Zwróć TYLKO poprawny JSON array, bez markdown, bez tekstu."},
                    ],
                    temperature=0.0,
                    max_tokens=16000,
                )
                raw2 = resp2.choices[0].message.content.strip()
                if raw2.startswith("```"):
                    raw2 = raw2.split("\n", 1)[1] if "\n" in raw2 else raw2[3:]
                    if raw2.endswith("```"):
                        raw2 = raw2[:-3]
                parsed2 = json.loads(raw2)
                results.extend(parsed2)
            except Exception as e2:
                st.error(f"Batch {idx+1}: nie udało się naprawić — {e2}")
                for item in batch:
                    kw = item if isinstance(item, str) else item.get("keyword", "?")
                    results.append({"keyword": kw, "_error": str(e2)})
        except Exception as e:
            st.error(f"Batch {idx+1}: błąd API — {e}")
            for item in batch:
                kw = item if isinstance(item, str) else item.get("keyword", "?")
                results.append({"keyword": kw, "_error": str(e)})

        if idx < len(batches) - 1:
            time.sleep(0.5)

    if placeholder:
        placeholder.progress(1.0, text="Gotowe!")
    return results


def process_serp_data(serp_df, main_df, domain_check):
    serp_df.columns = serp_df.columns.str.strip().str.lower()
    results = {}

    for kw in main_df["Keyword"].unique():
        kw_lower = kw.lower().strip()
        kw_data = serp_df[serp_df["keyword"].str.lower().str.strip() == kw_lower]

        if kw_data.empty:
            results[kw] = {
                "Pos_SEO_Explorer": None,
                "URL_best_Explorer": None,
                "Cannibalization_URLs": None,
                "Cannibalization_flag": "NIE",
                "SERP_features": None,
                "MM_in_SERP": "NIE",
                "TOP3_organic_URLs": None,
            }
            continue

        organic = kw_data[kw_data["type"] == "organic"].sort_values("rank_group")
        mm_organic = organic[organic["domain"].str.contains(domain_check, case=False, na=False)]

        pos = int(mm_organic["rank_group"].iloc[0]) if not mm_organic.empty else None
        url_best = mm_organic["url"].iloc[0] if not mm_organic.empty else None

        if len(mm_organic) > 1:
            parts = [f"{row['url']} [{int(row['rank_group'])}]" for _, row in mm_organic.iterrows()]
            cann_urls = " | ".join(parts)
            cann_flag = "TAK"
        elif len(mm_organic) == 1:
            cann_urls = f"{mm_organic['url'].iloc[0]} [{int(mm_organic['rank_group'].iloc[0])}]"
            cann_flag = "NIE"
        else:
            cann_urls = None
            cann_flag = "NIE"

        features = kw_data["type"].unique().tolist()
        serp_features = " | ".join(sorted(features))

        mm_anywhere = kw_data[kw_data["domain"].str.contains(domain_check, case=False, na=False)]
        mm_in_serp = "TAK" if not mm_anywhere.empty else "NIE"

        top3 = organic.head(3)
        if not top3.empty:
            parts = [f"{row['url']} [{int(row['rank_group'])}]" for _, row in top3.iterrows() if pd.notna(row['url'])]
            top3_str = " | ".join(parts)
        else:
            top3_str = None

        results[kw] = {
            "Pos_SEO_Explorer": pos,
            "URL_best_Explorer": url_best,
            "Cannibalization_URLs": cann_urls,
            "Cannibalization_flag": cann_flag,
            "SERP_features": serp_features,
            "MM_in_SERP": mm_in_serp,
            "TOP3_organic_URLs": top3_str,
        }

    return results


def process_site_product_data(site_df, main_df):
    site_df.columns = site_df.columns.str.strip().str.lower()
    results = {}
    for kw in main_df["Keyword"].unique():
        kw_lower = kw.lower().strip()
        kw_data = site_df[site_df["keyword"].str.lower().str.strip() == kw_lower]
        organic = kw_data[kw_data["type"] == "organic"].sort_values("rank_group")

        if organic.empty:
            results[kw] = {"Has_products": "NIE", "Product_URLs": None}
        else:
            urls = organic["url"].dropna().head(5).tolist()
            results[kw] = {
                "Has_products": "TAK",
                "Product_URLs": " | ".join(urls) if urls else None,
            }
    return results


def process_site_general_data(site_df, main_df, config):
    site_df.columns = site_df.columns.str.strip().str.lower()
    domain = config["domain"]
    product_path = config["product_path"]
    category_path = config["category_path"]
    content_path = config["content_path"]
    results = {}

    for kw in main_df["Keyword"].unique():
        kw_lower = kw.lower().strip()
        kw_data = site_df[site_df["keyword"].str.lower().str.strip() == kw_lower]
        organic = kw_data[kw_data["type"] == "organic"].sort_values("rank_group")

        if organic.empty:
            results[kw] = {
                "Site_results_count": 0,
                "Site_TOP_URLs": None,
                "URL_types_found": None,
            }
            continue

        count = len(organic)
        urls = organic["url"].dropna().head(5).tolist()
        top_str = " | ".join(urls) if urls else None

        types_found = set()
        for u in urls:
            if product_path in u:
                types_found.add("product")
            elif category_path in u:
                types_found.add("category")
            elif content_path in u:
                types_found.add("content")
            else:
                types_found.add("other")
        types_str = " | ".join(sorted(types_found)) if types_found else None

        results[kw] = {
            "Site_results_count": count,
            "Site_TOP_URLs": top_str,
            "URL_types_found": types_str,
        }
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
    st.session_state.config["model"] = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    st.session_state.config["batch_size"] = st.slider("Batch size (fraz na request)", 5, 50, st.session_state.config["batch_size"])
    st.session_state.config["serp_domain_check"] = st.text_input("Domena do sprawdzenia w SERP", st.session_state.config["serp_domain_check"])

    st.divider()
    st.subheader("Prompty")
    prompt_choice = st.selectbox("Edytuj prompt:", list(st.session_state.prompts.keys()))
    st.session_state.prompts[prompt_choice] = st.text_area(
        f"Prompt: {prompt_choice}",
        st.session_state.prompts[prompt_choice],
        height=300,
    )

    st.divider()
    if st.button("🔄 Reset aplikacji", type="secondary"):
        for k in list(st.session_state.keys()):
            if k != "authenticated":
                del st.session_state[k]
        st.rerun()


# ── Main UI ──────────────────────────────────────────────────
st.title(f"🔍 SEO Category Analyzer — {st.session_state.config['client_name']}")

step_cols = st.columns(len(STEPS))
for i, label in enumerate(STEPS):
    with step_cols[i]:
        if i < st.session_state.step:
            st.success(label, icon="✅")
        elif i == st.session_state.step:
            st.info(label, icon="👉")
        else:
            st.empty()

st.divider()

# ── STEP 0: Import ──────────────────────────────────────────
if st.session_state.step == 0:
    st.header("1. Import fraz kluczowych")
    st.write("Wgraj plik Excel z kolumnami: **Keyword**, **Volume**")

    uploaded = st.file_uploader("Wybierz plik XLSX", type=["xlsx", "xls", "csv"], key="upload_keywords")

    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.write(f"Znaleziono **{len(df)}** fraz")
        st.dataframe(df.head(20), use_container_width=True)

        req_cols = ["Keyword", "Volume"]
        # Try case-insensitive match
        col_map = {}
        for rc in req_cols:
            for c in df.columns:
                if c.lower().strip() == rc.lower():
                    col_map[c] = rc
        df = df.rename(columns=col_map)

        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            st.error(f"Brak kolumn: {missing}. Dostępne: {list(df.columns)}")
        else:
            if st.button("✅ Importuj frazy", type="primary"):
                df = df[["Keyword", "Volume"]].copy()
                df = df.dropna(subset=["Keyword"])
                df["Keyword"] = df["Keyword"].astype(str).str.strip()
                st.session_state.df = df
                st.session_state.step = 1
                st.rerun()

# ── STEP 1: Relevance ───────────────────────────────────────
elif st.session_state.step == 1:
    st.header("2. Relevance check — czy fraza jest dla nas?")
    df = st.session_state.df
    st.write(f"**{len(df)}** fraz do sprawdzenia")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button("📥 Pobierz aktualny XLSX", export_xlsx(df), f"seo_step1_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.button("🤖 Uruchom relevance check", type="primary"):
        kw_list = df["Keyword"].tolist()
        progress = st.empty()
        results = call_openai_batch(
            st.session_state.prompts["relevant"],
            kw_list,
            st.session_state.config,
            progress,
        )
        res_map = {r["keyword"]: r for r in results if "keyword" in r}

        df["Relevant_for_MM"] = df["Keyword"].map(lambda k: res_map.get(k, {}).get("relevant", "TAK"))
        df["Rejection_reason"] = df["Keyword"].map(lambda k: res_map.get(k, {}).get("reason", ""))
        st.session_state.df = df
        st.session_state.step = 2
        st.rerun()

# ── STEP 2: Classification ──────────────────────────────────
elif st.session_state.step == 2:
    st.header("3. Klasyfikacja AI — funnel, intent, segment, brand")
    df = st.session_state.df
    relevant_count = (df["Relevant_for_MM"] == "TAK").sum()
    rejected_count = (df["Relevant_for_MM"] == "NIE").sum()
    st.write(f"✅ Relewantnych: **{relevant_count}** | ❌ Odrzuconych: **{rejected_count}**")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button("📥 Pobierz aktualny XLSX", export_xlsx(df), f"seo_step2_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.button("🤖 Uruchom klasyfikację", type="primary"):
        to_classify = df[df["Relevant_for_MM"] == "TAK"]["Keyword"].tolist()
        progress = st.empty()
        results = call_openai_batch(
            st.session_state.prompts["classify"],
            to_classify,
            st.session_state.config,
            progress,
        )
        res_map = {r["keyword"]: r for r in results if "keyword" in r}

        for col in ["L1_Funnel_stage", "L2_Intent", "L3_MM_Segment", "Brand_flag", "Brand"]:
            df[col] = df["Keyword"].map(lambda k: res_map.get(k, {}).get(col, ""))

        st.session_state.df = df
        st.session_state.step = 3
        st.rerun()

# ── STEP 3: SERP import ─────────────────────────────────────
elif st.session_state.step == 3:
    st.header("4. Import danych SERP (SEO Explorer)")
    df = st.session_state.df
    st.write(f"**{len(df)}** fraz w analizie")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button("📥 Pobierz aktualny XLSX", export_xlsx(df), f"seo_step3_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.info("Wgraj plik z SEO Explorer — pełny SERP snapshot (kolumny: keyword, type, rank_group, rank_absolute, domain, url, title)")
    uploaded = st.file_uploader("Plik SERP snapshot", type=["xlsx", "xls", "csv"], key="upload_serp")

    if uploaded:
        if uploaded.name.endswith(".csv"):
            serp_df = pd.read_csv(uploaded)
        else:
            serp_df = pd.read_excel(uploaded)
        st.write(f"Załadowano **{len(serp_df)}** wierszy SERP, **{serp_df['keyword'].nunique() if 'keyword' in [c.lower() for c in serp_df.columns] else '?'}** unikalnych fraz")
        st.dataframe(serp_df.head(10), use_container_width=True)

        if st.button("✅ Przetwórz dane SERP", type="primary"):
            results = process_serp_data(serp_df, df, st.session_state.config["serp_domain_check"])
            for col in ["Pos_SEO_Explorer", "URL_best_Explorer", "Cannibalization_URLs", "Cannibalization_flag", "SERP_features", "MM_in_SERP", "TOP3_organic_URLs"]:
                df[col] = df["Keyword"].map(lambda k: results.get(k, {}).get(col))
            st.session_state.df = df
            st.session_state.step = 4
            st.rerun()

# ── STEP 4: site:…/product/ ─────────────────────────────────
elif st.session_state.step == 4:
    st.header("5. Import site:…/product/")
    df = st.session_state.df
    cfg = st.session_state.config
    st.write(f"Wgraj plik z wynikami: **fraza site:{cfg['domain']}{cfg['product_path']}**")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button("📥 Pobierz aktualny XLSX", export_xlsx(df), f"seo_step4_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    uploaded = st.file_uploader("Plik site:…/product/", type=["xlsx", "xls", "csv"], key="upload_product")

    if uploaded:
        if uploaded.name.endswith(".csv"):
            prod_df = pd.read_csv(uploaded)
        else:
            prod_df = pd.read_excel(uploaded)
        st.write(f"Załadowano **{len(prod_df)}** wierszy")
        st.dataframe(prod_df.head(10), use_container_width=True)

        if st.button("✅ Przetwórz dane produktów", type="primary"):
            results = process_site_product_data(prod_df, df)
            for col in ["Has_products", "Product_URLs"]:
                df[col] = df["Keyword"].map(lambda k: results.get(k, {}).get(col))
            st.session_state.df = df
            st.session_state.step = 5
            st.rerun()

# ── STEP 5: Products match intent ───────────────────────────
elif st.session_state.step == 5:
    st.header("6. Products match intent — AI sprawdza dopasowanie")
    df = st.session_state.df
    has_products = df[df["Has_products"] == "TAK"]
    st.write(f"**{len(has_products)}** fraz z produktami do sprawdzenia")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button("📥 Pobierz aktualny XLSX", export_xlsx(df), f"seo_step5_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.button("🤖 Uruchom products match intent", type="primary"):
        to_check = []
        for _, row in has_products.iterrows():
            to_check.append({"keyword": row["Keyword"], "product_urls": row.get("Product_URLs", "")})

        if to_check:
            progress = st.empty()
            results = call_openai_batch(
                st.session_state.prompts["products_match"],
                to_check,
                st.session_state.config,
                progress,
            )
            res_map = {r["keyword"]: r.get("match", "NIE") for r in results if "keyword" in r}
            df["Products_match_intent"] = df["Keyword"].map(lambda k: res_map.get(k, ""))
        else:
            df["Products_match_intent"] = ""

        st.session_state.df = df
        st.session_state.step = 6
        st.rerun()

# ── STEP 6: site:domain ─────────────────────────────────────
elif st.session_state.step == 6:
    st.header("7. Import site:domain (ogólne pokrycie)")
    df = st.session_state.df
    cfg = st.session_state.config
    st.write(f"Wgraj plik z wynikami: **fraza site:{cfg['domain']}**")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button("📥 Pobierz aktualny XLSX", export_xlsx(df), f"seo_step6_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    uploaded = st.file_uploader("Plik site:domain", type=["xlsx", "xls", "csv"], key="upload_site")

    if uploaded:
        if uploaded.name.endswith(".csv"):
            site_df = pd.read_csv(uploaded)
        else:
            site_df = pd.read_excel(uploaded)
        st.write(f"Załadowano **{len(site_df)}** wierszy")
        st.dataframe(site_df.head(10), use_container_width=True)

        if st.button("✅ Przetwórz dane site:domain", type="primary"):
            results = process_site_general_data(site_df, df, st.session_state.config)
            for col in ["Site_results_count", "Site_TOP_URLs", "URL_types_found"]:
                df[col] = df["Keyword"].map(lambda k: results.get(k, {}).get(col))
            st.session_state.df = df
            st.session_state.step = 7
            st.rerun()

# ── STEP 7: Content match intent ────────────────────────────
elif st.session_state.step == 7:
    st.header("8. Content match intent — AI sprawdza dopasowanie stron")
    df = st.session_state.df
    has_content = df[df["Site_results_count"].fillna(0).astype(int) > 0] if "Site_results_count" in df.columns else pd.DataFrame()
    st.write(f"**{len(has_content)}** fraz z pokryciem do sprawdzenia")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button("📥 Pobierz aktualny XLSX", export_xlsx(df), f"seo_step7_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.button("🤖 Uruchom content match intent", type="primary"):
        to_check = []
        for _, row in has_content.iterrows():
            to_check.append({"keyword": row["Keyword"], "site_urls": row.get("Site_TOP_URLs", "")})

        if to_check:
            progress = st.empty()
            results = call_openai_batch(
                st.session_state.prompts["content_match"],
                to_check,
                st.session_state.config,
                progress,
            )
            res_map = {r["keyword"]: r.get("match", "NIE") for r in results if "keyword" in r}
            df["Content_match_intent"] = df["Keyword"].map(lambda k: res_map.get(k, ""))
        else:
            df["Content_match_intent"] = ""

        st.session_state.df = df
        st.session_state.step = 8
        st.rerun()

# ── STEP 8: Final recommendations ───────────────────────────
elif st.session_state.step == 8:
    st.header("9. Rekomendacje AI — Action type + detail")
    df = st.session_state.df
    relevant = df[df["Relevant_for_MM"] == "TAK"]
    st.write(f"**{len(relevant)}** relewantnych fraz do analizy")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button("📥 Pobierz aktualny XLSX", export_xlsx(df), f"seo_step8_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.button("🤖 Uruchom rekomendacje", type="primary"):
        to_analyze = []
        cols_for_ai = ["Keyword", "Volume", "L1_Funnel_stage", "L2_Intent", "L3_MM_Segment",
                       "Brand_flag", "Brand", "Pos_SEO_Explorer", "Cannibalization_flag",
                       "SERP_features", "MM_in_SERP", "Has_products", "Products_match_intent",
                       "Site_results_count", "URL_types_found", "Content_match_intent"]
        available_cols = [c for c in cols_for_ai if c in relevant.columns]

        for _, row in relevant.iterrows():
            item = {c: (str(row[c]) if pd.notna(row[c]) else "") for c in available_cols}
            to_analyze.append(item)

        if to_analyze:
            progress = st.empty()
            results = call_openai_batch(
                st.session_state.prompts["action"],
                to_analyze,
                st.session_state.config,
                progress,
            )
            res_map = {r["keyword"]: r for r in results if "keyword" in r}

            for col in ["Action_type", "Action_detail", "Target_URL_suggested"]:
                df[col] = df["Keyword"].map(lambda k: res_map.get(k, {}).get(col, ""))

        st.session_state.df = df
        st.session_state.step = 9
        st.rerun()

# ── STEP 9: Final export ────────────────────────────────────
elif st.session_state.step == 9:
    st.header("10. Eksport finalny")
    df = st.session_state.df
    st.success(f"Analiza zakończona! **{len(df)}** fraz, **{len(df.columns)}** kolumn")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Relewantne frazy", (df["Relevant_for_MM"] == "TAK").sum())
        if "Has_products" in df.columns:
            st.metric("Z produktami", (df["Has_products"] == "TAK").sum())
    with col2:
        st.metric("Odrzucone", (df["Relevant_for_MM"] == "NIE").sum())
        if "Cannibalization_flag" in df.columns:
            st.metric("Kanibalizacja", (df["Cannibalization_flag"] == "TAK").sum())

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 Pobierz finalny XLSX",
        export_xlsx(df),
        f"seo_analysis_{st.session_state.config['client_name'].lower().replace(' ', '_')}_{date.today()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    if "Action_type" in df.columns:
        st.subheader("Rozkład rekomendacji")
        action_counts = df[df["Action_type"] != ""]["Action_type"].value_counts()
        st.bar_chart(action_counts)
