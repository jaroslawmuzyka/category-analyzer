import re

with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update init_state
text = text.replace('"session_total_cost": 0.0, "last_cost_info": ""}', '"session_total_cost": 0.0, "last_cost_info": "", "sitemaps": {}}')

# 2. Update STEPS
old_steps = 'STEPS = ["Import fraz", "Relevance", "Klasyfikacja", "SERP snapshot",\n         "site:…/product/", "Product match", "site:domain", "Content match",\n         "Video analysis", "Rekomendacje", "Eksport"]'
new_steps = 'STEPS = ["Import fraz", "Relevance", "Klasyfikacja", "Sitemap XML", "SERP snapshot",\n         "site:…/product/", "Product match", "site:domain", "Content match",\n         "Video analysis", "Rekomendacje", "Eksport"]'
text = text.replace(old_steps, new_steps)

# 3. Update all CS == and Step labels from top to bottom
# Export -> Krok 12
text = text.replace('CS == 10:', 'CS == 11:')
text = text.replace('Krok 11 · Eksport', 'Krok 12 · Eksport')

text = text.replace('CS == 9:', 'CS == 10:')
text = text.replace('Krok 10 · Rekomendacje', 'Krok 11 · Rekomendacje')
text = text.replace('nav_buttons(9)', 'nav_buttons(10)')
text = text.replace('st.session_state.step = 10', 'st.session_state.step = 11')

text = text.replace('CS == 8:', 'CS == 9:')
text = text.replace('Krok 9 · Video', 'Krok 10 · Video')
text = text.replace('nav_buttons(8)', 'nav_buttons(9)')
text = text.replace('st.session_state.step = 9', 'st.session_state.step = 10')

text = text.replace('CS == 7:', 'CS == 8:')
text = text.replace('Krok 8 · Content', 'Krok 9 · Content')
text = text.replace('nav_buttons(7)', 'nav_buttons(8)')
text = text.replace('st.session_state.step = 8', 'st.session_state.step = 9')

text = text.replace('CS == 6:', 'CS == 7:')
text = text.replace('Krok 7 · Import site:domain', 'Krok 8 · Import site:domain')
text = text.replace('nav_buttons(6)', 'nav_buttons(7)')
text = text.replace('st.session_state.step = 7', 'st.session_state.step = 8')

text = text.replace('CS == 5:', 'CS == 6:')
text = text.replace('Krok 6 · Products', 'Krok 7 · Products')
text = text.replace('nav_buttons(5)', 'nav_buttons(6)')
text = text.replace('st.session_state.step = 6', 'st.session_state.step = 7')

text = text.replace('CS == 4:', 'CS == 5:')
text = text.replace('Krok 5 · Import site', 'Krok 6 · Import site')
text = text.replace('nav_buttons(4)', 'nav_buttons(5)')
text = text.replace('st.session_state.step = 5', 'st.session_state.step = 6')

text = text.replace('CS == 3:', 'CS == 4:')
text = text.replace('Krok 4 · Import SERP', 'Krok 5 · Import SERP')
text = text.replace('nav_buttons(3)', 'nav_buttons(4)')
text = text.replace('st.session_state.step = 4', 'st.session_state.step = 5')

# 4. Insert new CS == 3
sitemap_tab = """# ── STEP 3: Sitemap XML ───────────────────────────────────
elif CS == 3:
    st.header("Krok 4 · Wczytywanie i Mapowanie Sitemapy XML")
    df = st.session_state.df
    st.write("Wgraj pliki Sitemaps.xml dla poszczególnych stref (nieobowiązkowe, ale bardzo zalecane do mapowania architektury).")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_file = st.file_uploader("Kategorie (XML)", type=["xml"], key="sm_cat")
        cat_mod_file = st.file_uploader("Kategorie modeli (XML)", type=["xml"], key="sm_catmod")
    with col2:
        filt_file = st.file_uploader("Filtry (XML)", type=["xml"], key="sm_filt")
        cont_file = st.file_uploader("Content/Blog (XML)", type=["xml"], key="sm_cont")
    with col3:
        search_file = st.file_uploader("Searchlist (XML)", type=["xml"], key="sm_search")
        serv_file = st.file_uploader("Service (XML)", type=["xml"], key="sm_serv")

    def parse_sm(file):
        if not file: return []
        import re
        txt = file.read().decode('utf-8')
        return re.findall(r'<loc>(?:<!\\[CDATA\\[)?(.*?)(?:\\]\\]>)?</loc>', txt)

    if st.button("📥 Przetwórz Sitemapy", type="primary"):
        st.session_state.sitemaps = {
            "Kategorie": parse_sm(cat_file),
            "Kategorie modeli": parse_sm(cat_mod_file),
            "Filtry": parse_sm(filt_file),
            "Content poradnik": parse_sm(cont_file),
            "Lista wyszukiwania": parse_sm(search_file),
            "Serwis lokalny": parse_sm(serv_file),
        }
        st.success("Sitemapy wczytane poprawnie!")
        
    sitemaps = st.session_state.get("sitemaps", {})
    if any(sitemaps.values()):
        st.write("Statystyki wczytanych adresów:")
        for k, v in sitemaps.items():
            if v:
                st.write(f"- {k}: **{len(v)}** adresów")
            
        if st.button("🤖 Mapuj adresy z Sitemapy (Lokalnie, 0$)", type="primary"):
            from rapidfuzz import process
            
            # Mapowanie kolumn L3_MM_Segment na konkretne sitemapy
            segment_to_sm = {
                "Kategoria producenta": "Kategorie",
                "Kategoria akcesoriów": "Kategorie",
                "Kategoria wariantu": "Kategorie",
                "Kategoria modelu": "Kategorie modeli",
                "Filtr kategorii": "Filtry",
                "Content poradnik": "Content poradnik",
                "Content versus": "Content poradnik",
                "Listing tematyczny": "Lista wyszukiwania",
                "Listing cenowa": "Lista wyszukiwania",
                "Serwis lokalny": "Serwis lokalny"
            }
            
            def get_top_urls(kw, segment):
                sm_key = segment_to_sm.get(segment)
                if not sm_key or sm_key not in sitemaps or not sitemaps[sm_key]:
                    return ""
                urls = sitemaps[sm_key]
                kw_norm = kw.replace(" ", "-").lower() # normalize for URL matching
                # extract best 10 matches
                results = process.extract(kw_norm, urls, limit=10)
                return " | ".join(r[0] for r in results)
                
            df["Sitemap_TOP10"] = df.apply(lambda row: get_top_urls(row["Keyword"], str(row.get("L3_MM_Segment", ""))), axis=1)
            st.session_state.df = df
            st.success("Zamapowano Sitemapy!")
            st.session_state.step = 4
            st.rerun()
            
    nav_buttons(3)

"""
text = text.replace('# ── STEP 3: SERP snapshot ───────────────────────────────────', sitemap_tab + '\n# ── STEP 4: SERP snapshot ───────────────────────────────────')

# 5. Inject Sitemap_TOP10 into the AI prompt action target attributes (ai_cols)
text = text.replace('"Content_match_intent",\n               "Matching_category_URL", "Video_channel", "Video_format"]', '"Content_match_intent",\n               "Matching_category_URL", "Video_channel", "Video_format", "Sitemap_TOP10"]')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("SUCCESS")
