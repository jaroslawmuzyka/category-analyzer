import pandas as pd

try:
    df = pd.read_excel("../frazy poprzydzielane 8k.xlsx")
    
    with open("../analysis_output.txt", "w", encoding="utf-8") as f:
        f.write("Columns:\n")
        f.write(str(df.columns.tolist()) + "\n\n")
        
        if "L2_Intent" in df.columns and "L3_MM_Segment" in df.columns:
            f.write("L2_Intent counts:\n")
            f.write(str(df["L2_Intent"].value_counts()) + "\n\n")
            
            f.write("L3_MM_Segment counts:\n")
            f.write(str(df["L3_MM_Segment"].value_counts()) + "\n\n")
            
            f.write("Group by L2_Intent and L3_MM_Segment:\n")
            grouped = df.groupby(["L2_Intent", "L3_MM_Segment"]).size().reset_index(name='counts')
            f.write(grouped.to_string(index=False) + "\n\n")
            
            f.write("Sample keywords per L3_MM_Segment:\n")
            for segment in df["L3_MM_Segment"].dropna().unique():
                f.write(f"\n--- {segment} ---\n")
                samples = df[df["L3_MM_Segment"] == segment]["Keyword"].dropna().sample(n=min(5, len(df[df["L3_MM_Segment"] == segment]))).tolist()
                for s in samples:
                    f.write(f"- {s}\n")
        
        # Identify suspicious entries
        f.write("\nSuspicious mapping (L2 doesn't logically match L3?):\n")
        if "L2_Intent" in df.columns and "L3_MM_Segment" in df.columns:
            suspicious = df[
                ((df["L2_Intent"] == "Informational") & (~df["L3_MM_Segment"].str.contains("Content", na=False))) |
                ((df["L3_MM_Segment"].str.contains("Content", na=False)) & (df["L2_Intent"] != "Informational"))
            ]
            if len(suspicious) > 0:
                f.write(f"Found {len(suspicious)} rows where Intent=Informational but Segment isn't Content, or vice versa\n")
                samples = suspicious.head(10)[["Keyword", "L2_Intent", "L3_MM_Segment"]].to_string(index=False)
                f.write(samples + "\n")
            else:
                f.write("No obvious intent vs segment mismatches found.\n")
                
except Exception as e:
    with open("../analysis_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {e}")
