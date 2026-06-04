"""Self-contained HTML engineering report.

Renders a single portable HTML file with plots embedded as base64 PNGs, the
run configuration, the validation/safety summary, and a plain-language readout.
No external assets, so the report is shareable as one file.
"""
import base64
import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
from jinja2 import Template

_TEMPLATE = Template("""<!doctype html>
<html><head><meta charset="utf-8"><title>ThermaLoop — {{ title }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 980px; margin: 2rem auto; color: #1b2733; padding: 0 1rem; }
  h1 { font-size: 1.5rem; border-bottom: 3px solid #2a9d8f; padding-bottom: .3rem; }
  h2 { font-size: 1.1rem; color: #2a9d8f; margin-top: 2rem; }
  .meta { color: #667; font-size: .85rem; }
  .desc { background: #f4f8f7; border-left: 3px solid #2a9d8f;
          padding: .6rem 1rem; border-radius: 4px; }
  table { border-collapse: collapse; margin: .5rem 0; font-size: .9rem; }
  td, th { border: 1px solid #d9d9d9; padding: .35rem .7rem; text-align: left; }
  th { background: #f4f8f7; }
  .ok { color: #2a9d8f; font-weight: 600; }
  .warn { color: #d1495b; font-weight: 600; }
  img { max-width: 100%; border: 1px solid #eee; border-radius: 6px; margin: .5rem 0; }
  .foot { color: #889; font-size: .8rem; margin-top: 2rem;
          border-top: 1px solid #eee; padding-top: .6rem; }
  code { background: #f4f4f4; padding: .1rem .3rem; border-radius: 3px; }
</style></head><body>
<h1>ThermaLoop — {{ title }}</h1>
<p class="meta">Generated {{ timestamp }} · ThermaLoop v{{ version }} · simulation lab, not CFD</p>
{% if description %}<p class="desc">{{ description }}</p>{% endif %}

<h2>Summary</h2>
<table>
{% for k, v in summary %}<tr><th>{{ k }}</th><td>{{ v }}</td></tr>{% endfor %}
</table>
<p class="{{ verdict_class }}">{{ verdict }}</p>

{% for section in sections %}
<h2>{{ section.title }}</h2>
{% if section.note %}<p class="meta">{{ section.note }}</p>{% endif %}
<img src="data:image/png;base64,{{ section.img }}"/>
{% endfor %}

<p class="foot">Assumptions and limits: see <code>docs/ASSUMPTIONS.md</code>.
Steady-state behavior is calibrated to Khalili et al. (2024) and enforced by the
test suite. This report is an educational engineering analysis, not a
production or certification artifact.</p>
</body></html>""")


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_report(title, description, summary, sections,
                  verdict, verdict_ok=True, version="0.2.0"):
    """Render the HTML string. `sections` is a list of (title, fig, note)."""
    rendered_sections = [
        dict(title=t, img=_fig_to_b64(f), note=note)
        for (t, f, note) in sections
    ]
    return _TEMPLATE.render(
        title=title, description=description,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=version, summary=summary, sections=rendered_sections,
        verdict=verdict,
        verdict_class="ok" if verdict_ok else "warn",
    )


def write_report(path, **kwargs):
    html = render_report(**kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(html)
    return path
