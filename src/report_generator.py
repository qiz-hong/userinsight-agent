"""
Report Generator Module

Generates analysis reports combining tables and charts:
- Excel report with multiple sheets
- HTML summary report
- Console summary output
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """Generate analysis reports in various formats."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Excel Report ──────────────────────────────────────────

    def generate_excel(
        self,
        segments_df: pd.DataFrame = None,
        profiles_df: pd.DataFrame = None,
        importance_df: pd.DataFrame = None,
        metrics: dict = None,
        filename: str = "analysis_report.xlsx",
    ) -> str:
        """
        Generate an Excel report with multiple sheets.
        """
        path = self.output_dir / filename

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            if segments_df is not None and not segments_df.empty:
                segments_df.to_excel(writer, sheet_name="Segments", index=False)
            if profiles_df is not None and not profiles_df.empty:
                profiles_df.to_excel(writer, sheet_name="Profiles", index=False)
            if importance_df is not None and not importance_df.empty:
                importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
            if metrics:
                pd.DataFrame([metrics]).to_excel(writer, sheet_name="Metrics", index=False)

        print(f"  Excel report: {path}")
        return str(path)

    # ── HTML Report ───────────────────────────────────────────

    def generate_html_summary(
        self,
        summary: dict = None,
        chart_paths: list = None,
        filename: str = "report.html",
    ) -> str:
        """
        Generate an HTML summary report embedding charts.
        """
        html = self._build_html(summary or {}, chart_paths or [])
        path = self.output_dir / filename
        path.write_text(html, encoding="utf-8")
        print(f"  HTML report: {path}")
        return str(path)

    def _build_html(self, summary: dict, chart_paths: list) -> str:
        """Build a self-contained HTML page."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        chart_html = ""
        for p in chart_paths:
            name = Path(p).name
            chart_html += (
                f'    <div class="chart">\n'
                f'      <h3>{name}</h3>\n'
                f'      <img src="{name}" alt="{name}" style="max-width:100%;">\n'
                f"    </div>\n"
            )

        summary_html = ""
        for k, v in summary.items():
            if isinstance(v, dict):
                summary_html += f"<h3>{k}</h3><ul>"
                for sk, sv in v.items():
                    summary_html += f"<li><b>{sk}:</b> {sv}</li>"
                summary_html += "</ul>"
            else:
                summary_html += f"<p><b>{k}:</b> {v}</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UserInsight Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
         background: #f5f7fa; color: #333; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .chart {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 30px; }}
  ul {{ padding-left: 20px; }}
  li {{ margin: 4px 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>UserInsight — Analysis Report</h1>
  <p class="meta">Generated: {now}</p>

  <h2>Summary</h2>
  {summary_html}

  <h2>Charts</h2>
{chart_html}
</div>
</body>
</html>"""

    # ── Console Summary ───────────────────────────────────────

    def print_console_summary(self, summary: dict) -> None:
        """Print a formatted summary to the console."""
        print("\n" + "=" * 55)
        print("  ANALYSIS SUMMARY")
        print("=" * 55)
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"\n  [{key}]")
                for sk, sv in value.items():
                    print(f"    {sk}: {sv}")
            elif isinstance(value, pd.DataFrame):
                print(f"\n  [{key}]")
                print(value.to_string(index=False))
            else:
                print(f"  {key}: {value}")
        print("=" * 55 + "\n")
