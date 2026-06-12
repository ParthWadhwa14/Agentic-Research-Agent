import tempfile
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def is_data_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in DATA_EXTENSIONS


def analyze_data_file(file_path: str, output_dir: Optional[str | Path] = None, query: str = "") -> Dict[str, Any]:
    """Runs local CSV/Excel analysis and emits reusable chart image paths."""
    result: Dict[str, Any] = {"summary": "", "chart_paths": []}
    path = Path(file_path)
    if not path.exists():
        result["summary"] = f"[Data file missing: {file_path}]"
        return result

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "research_agent_matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        result["summary"] = f"[Data analysis dependencies missing: {exc}]"
        return result

    try:
        df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
        output = Path(output_dir or tempfile.mkdtemp(prefix="data_charts_"))
        output.mkdir(parents=True, exist_ok=True)

        numeric_df = df.select_dtypes(include="number")
        categorical_df = df.select_dtypes(exclude="number")
        missing = df.isna().sum().to_dict()

        summary_parts = [
            f"Local data analysis for: {path.name}",
            f"User question: {query or 'General analysis'}",
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"Columns: {', '.join(map(str, df.columns))}",
            f"Missing values by column: {missing}",
            "Preview:\n" + df.head(8).to_string(index=False),
        ]

        if not numeric_df.empty:
            summary_parts.append("Numeric summary statistics:\n" + numeric_df.describe().to_string())
            correlation = numeric_df.corr(numeric_only=True)
            if len(numeric_df.columns) >= 2:
                summary_parts.append("Correlation matrix:\n" + correlation.to_string())

            hist_path = output / f"{path.stem}_histograms.png"
            numeric_df.hist(figsize=(10, 7), bins=20)
            plt.suptitle("Numeric Distributions")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=160)
            plt.close("all")
            result["chart_paths"].append(str(hist_path))

            if len(numeric_df.columns) >= 2:
                corr_path = output / f"{path.stem}_correlation.png"
                plt.figure(figsize=(8, 6))
                plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
                plt.colorbar(label="Correlation")
                plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
                plt.yticks(range(len(correlation.columns)), correlation.columns)
                plt.title("Correlation Heatmap")
                plt.tight_layout()
                plt.savefig(corr_path, dpi=160)
                plt.close("all")
                result["chart_paths"].append(str(corr_path))

                scatter_path = output / f"{path.stem}_scatter.png"
                x_col, y_col = numeric_df.columns[:2]
                plt.figure(figsize=(8, 5))
                plt.scatter(numeric_df[x_col], numeric_df[y_col], alpha=0.75)
                plt.xlabel(str(x_col))
                plt.ylabel(str(y_col))
                plt.title(f"{y_col} vs {x_col}")
                plt.tight_layout()
                plt.savefig(scatter_path, dpi=160)
                plt.close("all")
                result["chart_paths"].append(str(scatter_path))

            outlier_notes: List[str] = []
            for column in numeric_df.columns:
                q1 = numeric_df[column].quantile(0.25)
                q3 = numeric_df[column].quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                outliers = numeric_df[(numeric_df[column] < q1 - 1.5 * iqr) | (numeric_df[column] > q3 + 1.5 * iqr)]
                if not outliers.empty:
                    outlier_notes.append(f"{column}: {len(outliers)} potential outliers")
            summary_parts.append("Outlier detection: " + ("; ".join(outlier_notes) if outlier_notes else "No strong IQR outliers found."))
        else:
            summary_parts.append("No numeric columns detected, so numeric charts/correlations were not created.")

        if not categorical_df.empty:
            useful_cats = [column for column in categorical_df.columns if 1 < categorical_df[column].nunique(dropna=True) <= 25]
            if useful_cats:
                cat_col = useful_cats[0]
                counts = categorical_df[cat_col].value_counts(dropna=False).head(12)
                summary_parts.append(f"Top categories for {cat_col}:\n" + counts.to_string())
                bar_path = output / f"{path.stem}_{cat_col}_counts.png"
                plt.figure(figsize=(9, 5))
                counts.plot(kind="bar")
                plt.title(f"Top {cat_col} Counts")
                plt.xlabel(str(cat_col))
                plt.ylabel("Count")
                plt.xticks(rotation=35, ha="right")
                plt.tight_layout()
                plt.savefig(bar_path, dpi=160)
                plt.close("all")
                result["chart_paths"].append(str(bar_path))

        summary_parts.append(
            "Analyst guidance: answer from the uploaded dataset first. Use charts where they clarify distributions, relationships, categories, or outliers."
        )
        result["summary"] = "\n\n".join(summary_parts)
        return result
    except Exception as exc:
        result["summary"] = f"[Data analysis failed for {file_path}: {exc}]"
        return result


def analyze_data_files(paths: Iterable[str], output_dir: Optional[str | Path] = None, query: str = "") -> Dict[str, Any]:
    summaries: List[str] = []
    chart_paths: List[str] = []
    for path in paths:
        analysis = analyze_data_file(path, output_dir=output_dir, query=query)
        if analysis.get("summary"):
            summaries.append(str(analysis["summary"]))
        chart_paths.extend([str(item) for item in analysis.get("chart_paths", [])])
    return {"summary": "\n\n---\n\n".join(summaries), "chart_paths": chart_paths}
