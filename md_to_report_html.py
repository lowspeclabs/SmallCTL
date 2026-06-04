#!/usr/bin/env python3
"""
md_to_report_html.py — Convert markdown reports to paginated HTML using report-template.html styling.

Usage:
    python3 md_to_report_html.py input.md [output.html]

If output is omitted, writes to input.md with .html extension.
"""

import re
import sys
import os

def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def md_inline(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return text

def parse_markdown(md_text):
    """Parse markdown into title, meta dict, and list of section dicts."""
    title_match = re.search(r'^# (.+)', md_text)
    title = title_match.group(1) if title_match else 'Report'

    meta = {}
    for line in md_text.split('\n'):
        m = re.match(r'^\*\*(.+?):\*\*\s*(.+)', line.strip())
        if m:
            meta[m.group(1)] = m.group(2)

    sections = []
    current = None
    lines = md_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        m = re.match(r'^## (.+)', stripped)
        if m:
            if current:
                sections.append(current)
            current = {'title': m.group(1), 'level': 2, 'content': []}
            i += 1
            continue

        m = re.match(r'^### (.+)', stripped)
        if m:
            if current:
                current['content'].append(('h3', m.group(1)))
            i += 1
            continue

        m = re.match(r'^#### (.+)', stripped)
        if m:
            if current:
                current['content'].append(('h4', m.group(1)))
            i += 1
            continue

        if stripped.startswith('|'):
            table_lines = [stripped]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('|'):
                table_lines.append(lines[j].strip())
                j += 1
            if current:
                current['content'].append(('table', table_lines))
            i = j
            continue

        if stripped.startswith('```'):
            code_lines = []
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('```'):
                code_lines.append(lines[j])
                j += 1
            if current:
                current['content'].append(('code', code_lines))
            i = j + 1
            continue

        if stripped.startswith('**') and stripped.endswith('**') and ':' not in stripped:
            text = stripped[2:-2]
            if current:
                current['content'].append(('p', f'<strong>{esc(text)}</strong>'))
            i += 1
            continue

        m = re.match(r'^\*\*(.+?)\*\*\s*(.*)', stripped)
        if m and current:
            current['content'].append(('p', f'<strong>{esc(m.group(1))}</strong> {esc(m.group(2))}'))
            i += 1
            continue

        m = re.match(r'^[-*]\s+(.+)', stripped)
        if m and current:
            current['content'].append(('li', m.group(1)))
            i += 1
            continue

        m = re.match(r'^\d+\.\s+(.+)', stripped)
        if m and current:
            current['content'].append(('oli', m.group(1)))
            i += 1
            continue

        if stripped and current:
            current['content'].append(('p', stripped))

        i += 1

    if current:
        sections.append(current)

    return title, meta, sections

def build_section_html(sec, kicker_num):
    """Build HTML body for a single section page."""
    body_html = ''
    in_ul = False
    in_ol = False

    for ctype, cdata in sec['content']:
        if ctype == 'h3':
            if in_ul:
                body_html += '</ul>'
                in_ul = False
            if in_ol:
                body_html += '</ol>'
                in_ol = False
            body_html += f'<h3>{esc(cdata)}</h3>'

        elif ctype == 'h4':
            if in_ul:
                body_html += '</ul>'
                in_ul = False
            if in_ol:
                body_html += '</ol>'
                in_ol = False
            body_html += f'<h4>{esc(cdata)}</h4>'

        elif ctype == 'p':
            if in_ul:
                body_html += '</ul>'
                in_ul = False
            if in_ol:
                body_html += '</ol>'
                in_ol = False
            if cdata.startswith('<strong>'):
                body_html += f'<p>{cdata}</p>'
            else:
                body_html += f'<p>{md_inline(esc(cdata))}</p>'

        elif ctype == 'li':
            if in_ol:
                body_html += '</ol>'
                in_ol = False
            if not in_ul:
                body_html += '<ul>'
                in_ul = True
            body_html += f'<li>{md_inline(esc(cdata))}</li>'

        elif ctype == 'oli':
            if in_ul:
                body_html += '</ul>'
                in_ul = False
            if not in_ol:
                body_html += '<ol>'
                in_ol = True
            body_html += f'<li>{md_inline(esc(cdata))}</li>'

        elif ctype == 'table':
            if in_ul:
                body_html += '</ul>'
                in_ul = False
            if in_ol:
                body_html += '</ol>'
                in_ol = False

            rows = []
            for tl in cdata:
                cells = [c.strip() for c in tl.split('|')]
                cells = [c for c in cells if c and not re.match(r'^[-:]+$', c)]
                if cells:
                    rows.append(cells)

            if rows:
                body_html += '<div class="table-wrap"><table>'
                body_html += '<thead><tr>'
                for h in rows[0]:
                    body_html += f'<th>{md_inline(esc(h))}</th>'
                body_html += '</tr></thead><tbody>'
                for row in rows[1:]:
                    body_html += '<tr>'
                    for cell in row:
                        body_html += f'<td>{md_inline(esc(cell))}</td>'
                    body_html += '</tr>'
                body_html += '</tbody></table></div>'

        elif ctype == 'code':
            if in_ul:
                body_html += '</ul>'
                in_ul = False
            if in_ol:
                body_html += '</ol>'
                in_ol = False
            code_text = '\n'.join(cdata)
            body_html += f'<pre>{esc(code_text)}</pre>'

    if in_ul:
        body_html += '</ul>'
    if in_ol:
        body_html += '</ol>'

    page_id = re.sub(r'[^a-z0-9-]', '-', sec['title'].lower()).strip('-')
    kicker = str(kicker_num).zfill(2)

    return f'''
      <section class="page" data-title="{esc(sec['title'])}" data-kicker="{kicker}" id="{page_id}">
        <div class="doc-card"><h2>{esc(sec['title'])}</h2>
{body_html}
</div>
      </section>
'''

def build_hero_html(title, meta, sections):
    """Build the overview/hero page from extracted metadata."""
    completed = meta.get('Completed', '')
    failed = meta.get('Failed', '')
    model = meta.get('Model', '')
    backend = meta.get('Backend', '')
    runtime = meta.get('Runtime', '')
    challenges = meta.get('Challenges', '')

    subtitle_parts = [p for p in [model, backend, runtime] if p]
    subtitle = ' · '.join(subtitle_parts) if subtitle_parts else ''

    # Try to infer metric cards from meta
    cards_html = ''
    if completed:
        num = completed.split('/')[0] if '/' in completed else completed
        cards_html += f'<div class="card"><div class="metric ok">{num}</div><div class="label">Completed / {completed.split("/")[-1] if "/" in completed else "total"}</div></div>'
    if failed:
        num = failed.split('/')[0] if '/' in failed else failed
        cards_html += f'<div class="card"><div class="metric danger">{num}</div><div class="label">Failed / {failed.split("/")[-1] if "/" in failed else "total"}</div></div>'
    if challenges:
        cards_html += f'<div class="card"><div class="metric">{challenges}</div><div class="label">Challenges</div></div>'

    # Generic eyebrow from first word of title or default
    eyebrow = title.split()[0] + ' Report' if title else 'Report'

    return f'''
      <section class="page active" data-title="Overview" data-kicker="01" id="overview">
        <div class="hero">
          <div class="hero-card">
            <div class="eyebrow">{esc(eyebrow)}</div>
            <h1>{esc(title)}</h1>
            {f'<p class="subtle" style="max-width:820px;font-size:18px;line-height:1.65">{esc(subtitle)}</p>' if subtitle else ''}
            <div class="grid cards" style="margin-top:28px">
              {cards_html}
            </div>
          </div>
        </div>
      </section>
'''

def build_full_html(title, meta, sections, source_name):
    """Assemble the complete HTML document."""
    pages_html = build_hero_html(title, meta, sections)

    page_num = 2
    for sec in sections:
        pages_html += build_section_html(sec, page_num)
        page_num += 1

    total_pages = page_num - 1

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{esc(title)}</title>
  <style>
:root {{
  color-scheme: dark;
  --bg: #0b0f14;
  --surface: #111821;
  --surface-2: #172230;
  --surface-3: #1d2a3a;
  --text: #e7edf5;
  --muted: #9fb0c7;
  --muted-2: #718096;
  --line: rgba(255,255,255,.10);
  --accent: #7c9cff;
  --accent-2: #29d3a3;
  --danger: #ff5d73;
  --warn: #ffcc66;
  --ok: #62e6a8;
  --shadow: 0 20px 60px rgba(0,0,0,.40);
  --radius: 22px;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  min-height: 100vh;
  font-family: var(--sans);
  background:
    radial-gradient(circle at 20% 0%, rgba(124,156,255,.25), transparent 30%),
    radial-gradient(circle at 90% 20%, rgba(41,211,163,.12), transparent 26%),
    var(--bg);
  color: var(--text);
}}
a {{ color: inherit; }}
.app {{ display: grid; grid-template-columns: 300px minmax(0, 1fr); min-height: 100vh; }}
aside {{
  position: sticky; top: 0; height: 100vh; overflow: auto;
  padding: 24px 18px;
  background: rgba(9, 14, 20, .78);
  border-right: 1px solid var(--line);
  backdrop-filter: blur(18px);
}}
.brand {{ padding: 12px 10px 20px; }}
.eyebrow {{ color: var(--accent-2); font-size: 12px; text-transform: uppercase; letter-spacing: .14em; font-weight: 800; }}
h1 {{ margin: 8px 0 8px; font-size: clamp(34px, 4vw, 62px); line-height: .95; letter-spacing: -.05em; }}
.brand h2 {{ margin: 8px 0 0; font-size: 21px; line-height: 1.05; letter-spacing: -.02em; }}
.subtle {{ color: var(--muted); }}
.small {{ font-size: 13px; }}
nav {{ display: grid; gap: 8px; margin-top: 18px; }}
.nav-btn {{
  width: 100%; border: 1px solid transparent; background: transparent; color: var(--muted);
  text-align: left; padding: 12px 13px; border-radius: 16px; cursor: pointer;
  font-weight: 700; display: flex; align-items: center; justify-content: space-between; gap: 12px;
  transition: .18s ease;
}}
.nav-btn:hover, .nav-btn.active {{ background: rgba(124,156,255,.12); color: var(--text); border-color: rgba(124,156,255,.22); }}
.nav-btn span:last-child {{ color: var(--muted-2); font-size: 12px; font-family: var(--mono); }}
main {{ min-width: 0; padding: 24px; }}
.topbar {{
  position: sticky; top: 0; z-index: 5; display: flex; align-items: center; justify-content: space-between;
  gap: 16px; margin: -24px -24px 22px; padding: 14px 24px;
  border-bottom: 1px solid var(--line); background: rgba(11,15,20,.72); backdrop-filter: blur(18px);
}}
.pager {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
button, .button {{
  border: 1px solid var(--line); background: rgba(255,255,255,.06); color: var(--text); border-radius: 999px;
  padding: 10px 14px; font-weight: 800; cursor: pointer; transition: .18s ease; text-decoration: none; display: inline-flex; gap: 8px; align-items: center;
}}
button:hover, .button:hover {{ transform: translateY(-1px); background: rgba(255,255,255,.10); }}
.page {{ display: none; max-width: 1280px; margin: 0 auto; animation: fade .22s ease; }}
.page.active {{ display: block; }}
@keyframes fade {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
.hero {{ min-height: calc(100vh - 96px); display: grid; align-content: center; gap: 26px; padding: 40px 0 70px; }}
.hero-card {{
  padding: clamp(24px, 5vw, 54px); border-radius: 34px; border: 1px solid var(--line);
  background: linear-gradient(145deg, rgba(23,34,48,.90), rgba(17,24,33,.72)); box-shadow: var(--shadow);
  position: relative; overflow: hidden;
}}
.hero-card::after {{ content: ''; position: absolute; inset: auto -20% -55% 45%; height: 420px; background: radial-gradient(circle, rgba(124,156,255,.22), transparent 62%); pointer-events: none; }}
.grid {{ display: grid; gap: 16px; }}
.grid.cards {{ grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }}
.two {{ grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }}
.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.card, .doc-card {{
  background: rgba(17,24,33,.82); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 18px; box-shadow: 0 12px 28px rgba(0,0,0,.18);
}}
.metric {{ font-size: 38px; font-weight: 900; line-height: 1; letter-spacing: -.04em; }}
.metric.danger {{ color: var(--danger); }}
.metric.ok {{ color: var(--ok); }}
.metric.warn {{ color: var(--warn); }}
.label {{ color: var(--muted); margin-top: 8px; font-size: 13px; font-weight: 700; }}
.section-title {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin: 12px 0 18px; }}
h2 {{ font-size: clamp(26px, 3vw, 42px); margin: 0; line-height: 1; letter-spacing: -.04em; }}
h3 {{ margin: 0 0 12px; font-size: 19px; }}
.doc-card h2 {{ font-size: clamp(24px, 2.4vw, 36px); margin: 0 0 18px; }}
.doc-card h3 {{ margin: 24px 0 12px; color: var(--text); font-size: 22px; letter-spacing: -.02em; }}
.doc-card h4 {{ margin: 20px 0 8px; color: var(--accent-2); font-size: 16px; text-transform: uppercase; letter-spacing: .08em; }}
.doc-card p {{ color: var(--muted); line-height: 1.7; }}
.doc-card li {{ color: var(--muted); line-height: 1.6; margin: 5px 0; }}
ul.clean {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }}
ul.clean li {{ padding: 12px 14px; border-radius: 15px; background: rgba(255,255,255,.045); border: 1px solid var(--line); color: var(--text); }}
.checklist {{ list-style: none; padding-left: 0; }}
.checklist li {{ display: flex; gap: 9px; align-items: flex-start; }}
.box {{ display: inline-grid; place-items: center; min-width: 16px; width: 16px; height: 16px; border-radius: 5px; border: 1px solid rgba(124,156,255,.35); margin-top: 4px; color: var(--accent-2); font-size: 11px; }}
code, .code {{ font-family: var(--mono); background: rgba(124,156,255,.12); border: 1px solid rgba(124,156,255,.18); border-radius: 8px; padding: .13rem .38rem; color: #cbd8ff; }}
.table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: var(--radius); background: rgba(17,24,33,.72); margin: 14px 0; }}
table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
th, td {{ padding: 13px 14px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ position: sticky; top: 0; background: #121a24; z-index: 1; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
tr:hover td {{ background: rgba(124,156,255,.055); }}
pre {{
  max-height: 620px; overflow: auto; white-space: pre-wrap; word-break: break-word;
  background: rgba(4,8,13,.75); border: 1px solid rgba(255,255,255,.10);
  border-radius: 16px; padding: 14px; color: #c9d7ea; font-family: var(--mono); font-size: 12px; line-height: 1.5;
}}
.footer-note {{ color: var(--muted-2); font-size: 12px; padding: 28px 0 10px; }}
@media (max-width: 980px) {{
  .app {{ grid-template-columns: 1fr; }}
  aside {{ position: relative; height: auto; }}
  nav {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .grid.cards, .two, .three {{ grid-template-columns: 1fr; }}
  main {{ padding: 16px; }}
  .topbar {{ margin: -16px -16px 18px; padding: 12px 16px; }}
}}
@media print {{
  body {{ background: #fff; color: #111; }}
  aside, .topbar {{ display: none !important; }}
  .app {{ display: block; }}
  main {{ padding: 0; }}
  .page {{ display: block !important; page-break-after: always; max-width: none; }}
  .card, .hero-card, .doc-card {{ box-shadow: none; background: #fff; color: #111; border-color: #ddd; }}
  .subtle, .label {{ color: #444; }}
  .doc-card p, .doc-card li {{ color: #222; }}
}}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <div class="eyebrow">{esc(title.split()[0] if title else 'Report')} Report</div>
        <h2>{esc(title)}</h2>
        <p class="subtle small">Generated from {esc(source_name)}</p>
      </div>
      <nav id="nav"></nav>
    </aside>
    <main>
      <div class="topbar">
        <div class="subtle small"><span id="pageCounter">Page 1 / {total_pages}</span></div>
        <div class="pager">
          <button id="prevBtn" type="button">← Previous</button>
          <button id="nextBtn" type="button">Next →</button>
          <button type="button" onclick="window.print()">Print / PDF</button>
        </div>
      </div>

{pages_html}

      <p class="footer-note">Generated from {esc(source_name)}</p>
    </main>
  </div>

  <script>
    const pages = Array.from(document.querySelectorAll('.page'));
    const nav = document.getElementById('nav');
    let current = 0;

    pages.forEach((page, i) => {{
      const btn = document.createElement('button');
      btn.className = 'nav-btn' + (i === 0 ? ' active' : '');
      btn.type = 'button';
      btn.innerHTML = `<span>${{page.dataset.title}}</span><span>${{page.dataset.kicker}}</span>`;
      btn.addEventListener('click', () => showPage(i));
      nav.appendChild(btn);
    }});

    function showPage(i) {{
      current = (i + pages.length) % pages.length;
      pages.forEach((p, idx) => p.classList.toggle('active', idx === current));
      Array.from(nav.children).forEach((b, idx) => b.classList.toggle('active', idx === current));
      document.getElementById('pageCounter').textContent = `Page ${{current + 1}} / ${{pages.length}}`;
      history.replaceState(null, '', '#' + pages[current].id);
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}

    document.getElementById('prevBtn').addEventListener('click', () => showPage(current - 1));
    document.getElementById('nextBtn').addEventListener('click', () => showPage(current + 1));
    window.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowRight') showPage(current + 1);
      if (e.key === 'ArrowLeft') showPage(current - 1);
    }});
    const hashIndex = pages.findIndex(p => '#' + p.id === location.hash);
    if (hashIndex >= 0) showPage(hashIndex);
  </script>
</body>
</html>'''

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 md_to_report_html.py input.md [output.html]")
        print("       If output is omitted, writes to input.md with .html extension")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = input_path.rsplit('.', 1)[0] + '.html'

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'r') as f:
        md_text = f.read()

    title, meta, sections = parse_markdown(md_text)
    source_name = os.path.basename(input_path)
    html = build_full_html(title, meta, sections, source_name)

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Written {len(sections) + 1} pages to {output_path}")

if __name__ == '__main__':
    main()
