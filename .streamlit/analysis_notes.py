import pandas as pd

try:
    df = pd.read_excel("../seo_analysis_mediamarkt_2026-04-09 (6).xlsx")
    
    with open("notes_analysis.txt", "w", encoding="utf-8") as f:
        f.write("Columns:\n")
        f.write(str(df.columns.tolist()) + "\n\n")
        
        # Ostatnie 4 kolumny
        last_cols = df.columns[-4:].tolist()
        df_subset = df[["Keyword"] + last_cols].dropna(how="all", subset=last_cols)
        
        # Filtrujemy wiersze, w których ostatnie kolumny nie są wszystkie NaN
        # Odrzuć widma (NaN keyword)
        df_subset = df_subset.dropna(subset=["Keyword"])
        
        for idx, row in df_subset.head(20).iterrows():
            f.write(f"--- Row {idx+1} ---\n")
            f.write(str(row.to_dict()) + "\n\n")
            
except Exception as e:
    with open("notes_analysis.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {str(e)}")
