# Google Search Console + Bing setup (1bit.monster)

## Why
GitHub repo analytics showed ~7 unique visitors arriving from Google in 14 days.
GSC/Bing tell you *exactly* which search queries surface 1bit.monster, how many
impressions/clicks you get, and which pages rank. This is the definitive
"am I found naturally" answer for the website itself.

## Google Search Console (2 minutes, needs your Google account)
1. Go to https://search.google.com/search-console → "Add property" → **Domain** → enter `1bit.monster`
   (DNS TXT verification: add the TXT record at Namecheap → DNS. Namecheap: your domain uses
   `dns1/2.registrar-servers.com`, so add the TXT record under "Advanced DNS".)
2. OR use URL-prefix property `https://1bit.monster/` and verify with the **HTML file** method:
   download the `google*.html` verification file into `site/` and I will commit + deploy it
   (the deploy workflow ships it to GitHub Pages automatically).
3. In GSC: **Sitemaps** → submit `https://1bit.monster/sitemap.xml`
4. After a few days, check **Performance** (queries/impressions/clicks) and **Indexing**.

## Bing Webmaster Tools (optional, 1 minute)
1. https://www.bing.com/webmasters → import from Google Search Console (or add site manually)
2. Submit the same sitemap.

## Housekeeping already verified
- `robots.txt`: `Allow: /` + sitemap reference — OK
- `sitemap.xml`: 31 URLs, all resolve to real files in `site/` — OK
- Every page has a `<title>`; index has meta description — OK
- Pages use `.html` suffixes with descriptive titles (`1bit.MONSTER: Blog`, etc.) — OK

## If you want the "Google will re-crawl now" nudge
Once GSC is verified you can use the URL Inspection tool → "Request indexing"
for the homepage and sitemap — no code changes needed.
