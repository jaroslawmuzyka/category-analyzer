from rapidfuzz import process, fuzz

kw1 = "iphone-14-pro"
kw2 = "iphone-14-pro-max"

urls = [
    "https://mediamarkt.pl/pl/category/apple-iphone-14-pro-70023.html",
    "https://mediamarkt.pl/pl/category/apple-iphone-14-pro-max-70024.html",
    "https://mediamarkt.pl/pl/category/apple-iphone-14-70021.html"
]

print("--- Default (WRatio) ---")
print("Query:", kw1)
print(process.extract(kw1, urls, limit=3))
print("Query:", kw2)
print(process.extract(kw2, urls, limit=3))

print("\n--- fuzz.ratio ---")
print("Query:", kw1)
print(process.extract(kw1, urls, limit=3, scorer=fuzz.ratio))
print("Query:", kw2)
print(process.extract(kw2, urls, limit=3, scorer=fuzz.ratio))

print("\n--- fuzz.partial_ratio ---")
print("Query:", kw1)
print(process.extract(kw1, urls, limit=3, scorer=fuzz.partial_ratio))
print("Query:", kw2)
print(process.extract(kw2, urls, limit=3, scorer=fuzz.partial_ratio))

print("\n--- fuzz.QRatio ---")
print("Query:", kw1)
print(process.extract(kw1, urls, limit=3, scorer=fuzz.QRatio))
print("Query:", kw2)
print(process.extract(kw2, urls, limit=3, scorer=fuzz.QRatio))

print("\n--- Custom logic ---")
# To fix this, we can extract just the last segment of the url, e.g. "apple-iphone-14-pro" and compare with it using ratio
import re
def get_path_score(kw, url):
    # Extract path portion
    match = re.search(r'/([^/]+)(?:-\d+)?(?:\.html)?/?$', url)
    if match:
        path = match.group(1).lower()
        return fuzz.ratio(kw, path)
    return fuzz.partial_ratio(kw, url)

for u in urls:
    print(u, kw1, get_path_score(kw1, u))
    print(u, kw2, get_path_score(kw2, u))
