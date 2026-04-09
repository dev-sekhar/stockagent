"""
ChartCompilerAgent
------------------
Single purpose: receive a list of CurveSpec objects from curve agents,
align their date axes, rebase price series to 100, build Plotly traces,
assemble layout, and save a self-contained interactive HTML file.

This agent owns ZERO data-fetching logic. All data arrives via CurveSpecs.
Separation of concerns:
  Curve agents  → fetch + clean data
  ChartCompiler → align + rebase + render + save

Subplot layout is inferred from the CurveSpec roles:
  Top panel (row 1):  stock, benchmark, sector curves
  Bottom panel (row 2): volume bars (normal) OR spread bars (arbitrage)

Crosshair cursor and unified hover tooltip are always enabled.
"""
import os
import webbrowser
from typing import Sequence

import pandas as pd

from tools.curve_spec import CurveSpec

# ── Colour palettes ───────────────────────────────────────────────────────────
_STOCK_PALETTE = [
    "#00b4d8",   # cyan
    "#e74c3c",   # red
    "#27ae60",   # green
    "#f0a500",   # amber
    "#9b59b6",   # purple
    "#1abc9c",   # teal
    "#e67e22",   # orange
    "#3498db",   # blue
]
_BENCHMARK_COLOR = "#a0a0c8"    # muted blue-grey
_SECTOR_COLOR    = "#f0a500"    # amber / gold
_INDEX_PALETTE   = ["#a0a0c8", "#c8a0a0", "#a0c8a8"]   # multiple benchmarks


def _rebase(values: list[float], base: float = 100.0) -> list[float]:
    if not values or values[0] == 0:
        return values
    f = base / values[0]
    return [v * f for v in values]


def _align_to_common(
    specs: list[CurveSpec],
) -> tuple[pd.DatetimeIndex, dict[str, list[float]]]:
    """
    Find the common date set across all valid specs and reindex each series.

    Returns
    -------
    common_dates : DatetimeIndex of shared dates
    aligned      : {spec.ticker+'|'+spec.role : reindexed close values}
    """
    valid = [s for s in specs if not s.error and s.dates]
    if not valid:
        return pd.DatetimeIndex([]), {}

    common: pd.DatetimeIndex | None = None
    for spec in valid:
        dt = pd.DatetimeIndex([pd.Timestamp(d) for d in spec.dates])
        common = dt if common is None else common.intersection(dt)

    if common is None or len(common) == 0:
        return pd.DatetimeIndex([]), {}

    aligned: dict[str, list[float]] = {}
    for spec in valid:
        dt     = pd.DatetimeIndex([pd.Timestamp(d) for d in spec.dates])
        series = pd.Series(spec.values, index=dt)
        ra     = series.reindex(common, method="ffill").dropna()
        key    = f"{spec.ticker}|{spec.role}"
        aligned[key] = ra.values.tolist()

    # Trim common to the intersection that all series actually have
    min_len = min(len(v) for v in aligned.values())
    if min_len < len(common):
        common = common[:min_len]
        aligned = {k: v[:min_len] for k, v in aligned.items()}

    return common, aligned


class ChartCompiler:
    """
    Assembles a list of CurveSpecs into a Plotly HTML chart.

    Public interface
    ----------------
    compile(specs, title, arbitrage, output_dir) -> str
    """

    def compile(
        self,
        specs:          Sequence[CurveSpec],
        title:          str = "",
        arbitrage:      bool = False,
        output_dir:     str | None = None,
        company_data:   dict | None = None,
        news:           list | None = None,
    ) -> str:
        """
        Build an interactive Plotly chart from the provided CurveSpecs.

        Parameters
        ----------
        specs        : all curve agent outputs (errored specs are silently skipped)
        title        : chart title (auto-generated if empty)
        arbitrage    : True → actual prices + spread panel; False → rebased + volume
        output_dir   : where to save the HTML file (default: cwd)
        company_data : optional dict from CompanyDataTool.fetch() — adds a
                       company-snapshot panel below the chart
        news         : optional list from NewsAgent.fetch() — added to panel

        Returns
        -------
        str : summary message with file path and per-stock stats
        """
        print(f"  🖼️  [ChartCompiler] Compiling chart from "
              f"{len(specs)} curve spec(s)…")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return "❌  plotly not installed. Run: pip install plotly"

        # ── Separate specs by role ─────────────────────────────────────────────
        stock_specs    = [s for s in specs if s.role == "stock"     and not s.error]
        bench_specs    = [s for s in specs if s.role == "benchmark" and not s.error]
        sector_specs   = [s for s in specs if s.role == "sector"    and not s.error]
        spread_specs   = [s for s in specs if s.role == "spread"    and not s.error]
        skipped        = [s for s in specs if s.error]

        for s in skipped:
            print(f"  ⚠️  [ChartCompiler] Skipping {s.name} ({s.role}): {s.error}")

        if not stock_specs:
            return "❌  No valid stock data to chart."

        # ── Date alignment across all price curves (not spread) ────────────────
        price_specs  = stock_specs + bench_specs + sector_specs
        common_dates, aligned = _align_to_common(price_specs)

        if len(common_dates) == 0:
            return "❌  No common trading dates found across the selected series."

        dates_str = [d.strftime("%Y-%m-%d") for d in common_dates]
        print(f"  📅 [ChartCompiler] {len(dates_str)} common trading days "
              f"({dates_str[0]} → {dates_str[-1]})")

        # ── Figure layout ──────────────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.03,
        )

        # ── Stock curves ───────────────────────────────────────────────────────
        for i, spec in enumerate(stock_specs):
            key    = f"{spec.ticker}|{spec.role}"
            closes = aligned.get(key, [])
            if not closes:
                continue

            color = spec.color or _STOCK_PALETTE[i % len(_STOCK_PALETTE)]
            hx    = color.lstrip("#")
            sr, sg, sb = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            fill_rgba  = f"rgba({sr},{sg},{sb},0.12)"

            pct_chg = spec.meta.get("pct_chg", 0)
            sign    = "▲" if pct_chg >= 0 else "▼"
            name    = f"{spec.name} ({sign}{abs(pct_chg):.1f}%)"

            if arbitrage:
                y_vals = [round(v, 4) for v in closes]
                hover  = f"<b>{spec.name}</b>: %{{y:.4f}}<extra></extra>"
                fill   = "none"
            else:
                rebased = _rebase(closes)
                y_vals  = [round(v, 2) for v in rebased]
                hover   = (
                    f"<b>{spec.name}</b>: %{{customdata[0]:.2f}}"
                    "  <span style='color:#888'>(idx %{y:.2f})</span>"
                    "<extra></extra>"
                )
                fill = "tozeroy"

            fig.add_trace(go.Scatter(
                x=dates_str, y=y_vals,
                name=name,
                line=dict(color=color, width=2.5),
                fill=fill,
                fillcolor=fill_rgba if not arbitrage else None,
                customdata=[[c, d] for c, d in zip(closes, dates_str)],
                hovertemplate=hover,
            ), row=1, col=1)

            # ── High / low annotations (single stock only) ─────────────────────
            if not arbitrage and len(stock_specs) == 1:
                high_date = spec.meta.get("high_date", "")
                low_date  = spec.meta.get("low_date",  "")
                high_val  = spec.meta.get("high",  0)
                low_val   = spec.meta.get("low",   0)

                rebased_vals = _rebase(closes)
                if high_date in dates_str:
                    hi = dates_str.index(high_date)
                    fig.add_annotation(
                        x=dates_str[hi], y=rebased_vals[hi],
                        text=f"▲ High {high_val:.2f}",
                        showarrow=True, arrowhead=2, ax=0, ay=-30,
                        arrowcolor="#f39c12",
                        font=dict(color="#f39c12", size=10),
                        xref="x1", yref="y1",
                    )
                if low_date in dates_str:
                    li = dates_str.index(low_date)
                    fig.add_annotation(
                        x=dates_str[li], y=rebased_vals[li],
                        text=f"▼ Low {low_val:.2f}",
                        showarrow=True, arrowhead=2, ax=0, ay=30,
                        arrowcolor="#3498db",
                        font=dict(color="#3498db", size=10),
                        xref="x1", yref="y1",
                    )

        # ── Benchmark index curves ─────────────────────────────────────────────
        if not arbitrage:
            for j, spec in enumerate(bench_specs):
                key  = f"{spec.ticker}|{spec.role}"
                vals = aligned.get(key, [])
                if not vals:
                    continue
                rebased = _rebase(vals)
                pct     = spec.meta.get("pct_chg", 0)
                sign    = "▲" if pct >= 0 else "▼"
                color   = spec.color or _INDEX_PALETTE[j % len(_INDEX_PALETTE)]
                fig.add_trace(go.Scatter(
                    x=dates_str, y=[round(v, 2) for v in rebased],
                    name=f"{spec.name} ({sign}{abs(pct):.1f}%)",
                    line=dict(color=color, width=1.4, dash="dash"),
                    customdata=vals,
                    hovertemplate=(
                        f"<b>{spec.name}</b>: %{{customdata:.2f}}"
                        "  <span style='color:#888'>(idx %{y:.2f})</span>"
                        "<extra></extra>"
                    ),
                ), row=1, col=1)

            # ── Sector index curves ────────────────────────────────────────────
            for spec in sector_specs:
                key  = f"{spec.ticker}|{spec.role}"
                vals = aligned.get(key, [])
                if not vals:
                    continue
                rebased = _rebase(vals)
                pct     = spec.meta.get("pct_chg", 0)
                sign    = "▲" if pct >= 0 else "▼"
                color   = spec.color or _SECTOR_COLOR
                fig.add_trace(go.Scatter(
                    x=dates_str, y=[round(v, 2) for v in rebased],
                    name=f"{spec.name} ({sign}{abs(pct):.1f}%)",
                    line=dict(color=color, width=1.4, dash="dashdot"),
                    customdata=vals,
                    hovertemplate=(
                        f"<b>{spec.name}</b>: %{{customdata:.2f}}"
                        "  <span style='color:#888'>(idx %{y:.2f})</span>"
                        "<extra></extra>"
                    ),
                ), row=1, col=1)

            # Baseline at 100
            fig.add_hline(y=100, row=1, col=1,
                          line=dict(color="#555577", width=0.8, dash="solid"))

        # ── Bottom panel: volume OR spread ─────────────────────────────────────
        if arbitrage and spread_specs:
            for spec in spread_specs:
                if spec.error or not spec.dates:
                    continue
                bar_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in spec.values]
                fig.add_trace(go.Bar(
                    x=spec.dates,
                    y=[round(v, 3) for v in spec.values],
                    name=spec.name,
                    marker_color=bar_colors,
                    hovertemplate="Spread: %{y:.3f}%<extra></extra>",
                ), row=2, col=1)
            fig.add_hline(y=0, row=2, col=1,
                          line=dict(color="#555577", width=0.8))
        else:
            # Volume bars for each stock (grouped)
            for i, spec in enumerate(stock_specs):
                if not spec.volumes:
                    continue
                key   = f"{spec.ticker}|{spec.role}"
                color = spec.color or _STOCK_PALETTE[i % len(_STOCK_PALETTE)]
                # Align volumes to common dates
                dt    = pd.DatetimeIndex([pd.Timestamp(d) for d in spec.dates])
                vseries = pd.Series(spec.volumes, index=dt)
                valigned = vseries.reindex(common_dates, method="ffill").fillna(0)
                fig.add_trace(go.Bar(
                    x=dates_str,
                    y=valigned.values.tolist(),
                    name=f"{spec.name} Vol",
                    marker_color=color,
                    marker_opacity=0.45,
                    hovertemplate=f"{spec.name} Vol: %{{y:,.0f}}<extra></extra>",
                ), row=2, col=1)

        # ── Auto-generate title if not provided ────────────────────────────────
        if not title:
            names = [s.name for s in stock_specs]
            title = " vs ".join(names)

        # ── Layout ────────────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(text=title, font=dict(color="#eeeeee", size=14), x=0.5),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.03,
                xanchor="left", x=0,
                font=dict(color="#cccccc", size=10),
                bgcolor="rgba(22,33,62,0.7)",
            ),
            hoverlabel=dict(bgcolor="#16213e", bordercolor="#555577",
                            font=dict(size=12, color="#eeeeee")),
            height=650,
            margin=dict(l=65, r=30, t=90, b=40),
            barmode="group",
        )

        # Crosshair spike lines
        crosshair = dict(
            showspikes=True, spikemode="across+toaxis", spikedash="dot",
            spikecolor="#aaaaaa", spikethickness=1, spikesnap="cursor",
        )
        fig.update_xaxes(
            **crosshair,
            showgrid=True, gridcolor="#2a2a4a", gridwidth=0.5,
            tickfont=dict(color="#aaaaaa"), rangeslider=dict(visible=False),
        )
        fig.update_yaxes(
            showspikes=True, spikedash="dot", spikecolor="#aaaaaa", spikethickness=1,
            showgrid=True, gridcolor="#2a2a4a", gridwidth=0.5,
            tickfont=dict(color="#aaaaaa"),
        )
        fig.update_yaxes(
            title_text="Price" if arbitrage else "Indexed to 100",
            title_font=dict(color="#aaaaaa"), row=1, col=1,
        )
        fig.update_yaxes(
            title_text="Spread (%)" if (arbitrage and spread_specs) else "Volume",
            title_font=dict(color="#aaaaaa"), row=2, col=1,
        )

        # ── Save + open ────────────────────────────────────────────────────────
        out_dir  = output_dir or os.getcwd()
        # derive filename from stock tickers in the specs
        safe = "_vs_".join(
            s.ticker.replace(".", "_").replace("&", "") for s in stock_specs
        )
        filename = f"{safe}_{dates_str[0]}_{dates_str[-1]}.html"
        filepath = os.path.join(out_dir, filename)

        # Build HTML — use to_html so we can inject the company panel
        chart_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        if company_data or news:
            panel = self._build_company_panel(company_data or {}, news or [])
            chart_html = chart_html.replace("</body>", panel + "\n</body>")

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(chart_html)

        webbrowser.open(f"file:///{filepath.replace(os.sep, '/')}")
        print(f"  💾 [ChartCompiler] Saved → {filepath}")

        # ── Summary ────────────────────────────────────────────────────────────
        lines = [f"📊 Chart → {filepath}"]
        for spec in stock_specs:
            p   = spec.meta.get("pct_chg", 0)
            o   = spec.meta.get("open",  0)
            c   = spec.meta.get("close", 0)
            sign = "▲" if p >= 0 else "▼"
            lines.append(f"   {spec.name}: {o:.2f} → {c:.2f}  ({sign}{abs(p):.2f}%)")

        return "\n".join(lines)

    # ── Company snapshot panel ────────────────────────────────────────────────

    @staticmethod
    def _build_company_panel(company_data: dict, news: list) -> str:
        """
        Build a styled HTML panel (dark-theme, matching the chart) that shows
        company profile, key metrics, financials, and curated news.
        Injected into the chart HTML just before </body>.
        """
        if not company_data and not news:
            return ""

        p   = company_data.get("profile",      {})
        f   = company_data.get("fundamentals", {})
        fin = company_data.get("financials",   {})
        cur = f.get("currency", "")
        err = company_data.get("error")

        # ── CSS ───────────────────────────────────────────────────────────────
        style = """
<style>
  .co-panel {
    background: #1a1a2e; color: #ddddee; font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px; padding: 24px 32px; max-width: 1200px; margin: 0 auto 32px;
  }
  .co-panel h2 { color: #e0e0ff; font-size: 18px; margin: 0 0 4px; }
  .co-panel .co-sub { color: #9090b8; font-size: 12px; margin: 0 0 16px; }
  .co-panel .co-desc { color: #aaaacc; font-size: 12px; margin: 0 0 20px;
    line-height: 1.6; max-width: 900px; }
  .co-grid { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
  .co-card {
    background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px;
    padding: 14px 18px; flex: 1; min-width: 220px;
  }
  .co-card h3 { color: #8888bb; font-size: 11px; text-transform: uppercase;
    letter-spacing: 1px; margin: 0 0 10px; }
  .co-table { width: 100%; border-collapse: collapse; }
  .co-table tr td { padding: 4px 0; vertical-align: top; }
  .co-table tr td:first-child { color: #8888aa; width: 55%; font-size: 12px; }
  .co-table tr td:last-child  { color: #eeeeff; font-weight: 500; font-size: 12px; }
  .co-news { background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px;
    padding: 14px 18px; }
  .co-news h3 { color: #8888bb; font-size: 11px; text-transform: uppercase;
    letter-spacing: 1px; margin: 0 0 10px; }
  .co-news ul { margin: 0; padding: 0; list-style: none; }
  .co-news li { padding: 8px 0; border-bottom: 1px solid #2a2a4a; }
  .co-news li:last-child { border-bottom: none; }
  .co-news .ndate { color: #6666aa; font-size: 11px; margin-right: 6px; }
  .co-news a { color: #8ab4f8; text-decoration: none; }
  .co-news a:hover { text-decoration: underline; }
  .co-news .npub  { color: #6666aa; font-size: 11px; font-style: italic; }
  .co-news .nsum  { color: #aaaacc; font-size: 12px; margin-top: 2px; }
</style>"""

        def v(val, fmt=None, suffix=""):
            if val is None:
                return '<span style="color:#555">N/A</span>'
            if fmt == "large":
                return _fmt_large(val, cur) + suffix
            if fmt == "pct":
                return f"{val:.1f}%" + suffix
            if fmt == "x":
                return f"{val:.1f}×"
            return str(round(val, 2)) + suffix

        # ── Header ────────────────────────────────────────────────────────────
        name    = p.get("name") or company_data.get("ticker", "")
        sector  = p.get("sector")  or ""
        ind     = p.get("industry") or ""
        country = p.get("country")  or ""
        exch    = p.get("exchange") or ""
        website = p.get("website")  or ""
        desc    = p.get("description") or ""
        web_lnk = f' · <a href="{website}" style="color:#8ab4f8">{website}</a>' if website else ""

        sub_parts = [x for x in [sector, ind, country, exch] if x]
        sub_line  = "  ·  ".join(sub_parts) + web_lnk

        if err:
            body = f'<p style="color:#e74c3c">⚠️ {err}</p>'
        else:
            # Metrics card
            metrics_rows = [
                ("Market Cap",       v(f.get("market_cap"),       "large")),
                ("Enterprise Value", v(f.get("enterprise_value"), "large")),
                ("P/E TTM / Fwd",    f'{v(f.get("pe_trailing"))} / {v(f.get("pe_forward"))}'),
                ("EPS TTM / Fwd",    f'{v(f.get("eps_trailing"))} / {v(f.get("eps_forward"))}'),
                ("52-Week Range",    f'{v(f.get("week_52_low"))} – {v(f.get("week_52_high"))}'),
                ("Beta",             v(f.get("beta"))),
                ("Dividend Yield",   v(f.get("dividend_yield_pct"), "pct")),
                ("Avg Vol (10d)",    v(f.get("avg_volume_10d"),  "large")),
            ]
            metrics_html = "\n".join(
                f"<tr><td>{k}</td><td>{val}</td></tr>"
                for k, val in metrics_rows
            )

            # Financials card
            fin_rows = [
                ("Revenue",          v(fin.get("revenue"),       "large")),
                ("Gross Profit",     v(fin.get("gross_profit"),  "large")),
                ("Net Income",       v(fin.get("net_income"),    "large")),
                ("Gross Margin",     v(fin.get("gross_margin_pct"),    "pct")),
                ("Net Margin",       v(fin.get("net_margin_pct"),      "pct")),
                ("ROE / ROA",        f'{v(fin.get("roe_pct"), "pct")} / {v(fin.get("roa_pct"), "pct")}'),
                ("Debt / Equity",    v(fin.get("debt_to_equity"), "x")),
                ("Free Cash Flow",   v(fin.get("free_cash_flow"), "large")),
                ("Revenue Growth",   v(fin.get("revenue_growth_pct"),  "pct")),
            ]
            fin_html = "\n".join(
                f"<tr><td>{k}</td><td>{val}</td></tr>"
                for k, val in fin_rows
            )

            body = f"""
<div class="co-grid">
  <div class="co-card">
    <h3>📊 Key Metrics</h3>
    <table class="co-table">{metrics_html}</table>
  </div>
  <div class="co-card">
    <h3>💰 Financials (TTM)</h3>
    <table class="co-table">{fin_html}</table>
  </div>
</div>"""

        # News section
        news_html = ""
        if news:
            items_html = ""
            for item in news[:5]:
                title = item.get("title", "")
                url   = item.get("url", "#")
                pub   = item.get("publisher", "")
                date  = item.get("date", "")
                summ  = item.get("summary", "")
                items_html += (
                    f'<li>'
                    f'<span class="ndate">{date}</span>'
                    f'<a href="{url}" target="_blank">{title}</a>'
                    f' <span class="npub">· {pub}</span>'
                    + (f'<div class="nsum">{summ}</div>' if summ else "")
                    + "</li>\n"
                )
            news_html = f"""
<div class="co-news">
  <h3>📰 Latest News</h3>
  <ul>{items_html}</ul>
</div>"""

        return f"""{style}
<div class="co-panel">
  <h2>{name} &nbsp;<span style="color:#6666aa;font-size:14px">({company_data.get('ticker','')})</span></h2>
  <p class="co-sub">{sub_line}</p>
  {f'<p class="co-desc">{desc}</p>' if desc else ""}
  {body}
  {news_html}
</div>"""


# ── Number formatter (shared with company_profile.py fallback) ───────────────

def _fmt_large(val, currency: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        v      = float(val)
        prefix = currency + " " if currency else ""
        if v >= 1e12: return f"{prefix}{v / 1e12:.2f}T"
        if v >= 1e9:  return f"{prefix}{v / 1e9:.2f}B"
        if v >= 1e6:  return f"{prefix}{v / 1e6:.2f}M"
        return f"{prefix}{round(v, 2):,}"
    except Exception:
        return str(val)
