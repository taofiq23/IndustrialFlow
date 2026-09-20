"""Builds every figure and LaTeX table in the paper from study_results/*.json.

    python -m src.industrialflow.study_figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
STUDY_DIR = ROOT / "study_results"
PAPER_DIR = ROOT / "paper"

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
RULE_STYLE = {
    "uncorrected (released)": ("#c44e52", "o", "-"),
    "Bonferroni": ("#4c72b0", "s", "--"),
    "Benjamini-Hochberg": ("#dd8452", "^", "-."),
    "Fisher combination": ("#55a868", "D", "-"),
    "2-of-3 columns (per-column 0.01)": ("#8172b3", "v", ":"),
    "2-of-3 columns (size-calibrated)": ("#937860", "P", "--"),
}
SHORT = {
    "temperature +0.25 sd": "T +0.25",
    "temperature +0.5 sd": "T +0.5",
    "temperature +1.0 sd": "T +1.0",
    "temperature sd x1.5": "T sd$\\times$1.5",
    "all columns +0.15 sd": "All +0.15",
    "all columns +0.25 sd": "All +0.25",
    "all columns +0.35 sd": "All +0.35",
}


def _load(name: str) -> dict:
    return json.loads((STUDY_DIR / name).read_text())


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def _scenario(rules: dict, label: str) -> dict:
    return next(s for s in rules["scenarios"] if s["scenario"] == label)


def fig_power(rules: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    panels = (
        (axes[0], "(a) Temperature-only drift", [("no drift", 0.0), ("temperature +0.25 sd", 0.25), ("temperature +0.5 sd", 0.5), ("temperature +1.0 sd", 1.0)]),
        (axes[1], "(b) Small drift in all three columns", [("no drift", 0.0), ("all columns +0.15 sd", 0.15), ("all columns +0.25 sd", 0.25), ("all columns +0.35 sd", 0.35)]),
    )
    for ax, title, points in panels:
        for name, (color, marker, ls) in RULE_STYLE.items():
            ys = [100 * _scenario(rules, label)["rules"][name]["rate"] for label, _ in points]
            ax.plot([x for _, x in points], ys, marker=marker, ls=ls, color=color, label=name, ms=4)
        ax.set_xlabel("Mean shift (standard deviations)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Batches flagged (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(PAPER_DIR / "fig_power.pdf")
    plt.close(fig)


def fig_reference(reference: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    data = [100 * np.array(e["uncorrected_per_reference_rates"]) for e in reference["by_reference_size"]]
    labels = [str(e["n_reference"]) for e in reference["by_reference_size"]]
    ax.boxplot(data, tick_labels=labels, showfliers=True, flierprops={"markersize": 2}, medianprops={"color": "#c44e52"})
    pooled = np.mean([e["rules"]["uncorrected (released)"]["pooled_alarm_rate"] for e in reference["by_reference_size"]]) * 100
    ax.axhline(pooled, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Reference sample size (rows)")
    ax.set_ylabel("False-alarm rate per reference (%)")
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig_reference.pdf")
    plt.close(fig)


def fig_autocorr(autocorr: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    phis = [r["phi"] for r in autocorr["rows"]]
    styles = {
        "naive": ("#c44e52", "o", "Naive KS"),
        "thinned": ("#4c72b0", "s", "Thinned"),
        "ess_adjusted": ("#55a868", "^", "ESS-adjusted"),
        "block_permutation": ("#8172b3", "D", "Block permutation"),
    }
    ax = axes[0]
    for key, (color, marker, label) in styles.items():
        ax.plot(phis, [100 * r["null"][key]["rate"] for r in autocorr["rows"]], marker=marker, color=color, label=label, ms=4)
    ax.axhline(100 * autocorr["alpha"], color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Lag-1 autocorrelation $\\phi$")
    ax.set_ylabel("False-alarm rate (%, log scale)")
    ax.set_title("(a) No drift", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for key, (color, marker, label) in styles.items():
        ax.plot(phis, [100 * r["power"]["shift_0.5"][key]["rate"] for r in autocorr["rows"]], marker=marker, color=color, label=label, ms=4)
    ax.set_xlabel("Lag-1 autocorrelation $\\phi$")
    ax.set_ylabel("Batches flagged (%)")
    ax.set_title("(b) Temperature +0.5 sd", fontsize=9)
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig_autocorr.pdf")
    plt.close(fig)


def table_null(rules: dict) -> None:
    null = _scenario(rules, "no drift")
    k = 3
    theory = {
        "uncorrected (released)": 1 - (1 - 0.01) ** k,
        "Bonferroni": None,
        "Benjamini-Hochberg": 0.01,
        "Fisher combination": 0.01,
        "2-of-3 columns (per-column 0.01)": 3 * 0.01**2 - 2 * 0.01**3,
        "2-of-3 columns (size-calibrated)": 0.01,
    }
    theory["Bonferroni"] = 1 - (1 - 0.01 / k) ** k
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"\textbf{Alarm rule} & \textbf{Batches flagged (\%) [95\% CI]} & \textbf{Theory (\%)} \\", r"\midrule"]
    for name, cell in null["rules"].items():
        lines.append(f"{name} & {_pct(cell['rate'], 2)} [{_pct(cell['ci'][0], 2)}, {_pct(cell['ci'][1], 2)}] & {_pct(theory[name], 2)} \\\\")
    lines += [r"\midrule"]
    for column, cell in null["per_column_rejection"].items():
        lines.append(f"Single column: {column.replace('_', ' ')} & {_pct(cell['rate'], 2)} [{_pct(cell['ci'][0], 2)}, {_pct(cell['ci'][1], 2)}] & 1.00 \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER_DIR / "tab_null.tex").write_text("\n".join(lines) + "\n")


def table_power(rules: dict) -> None:
    labels = list(SHORT)
    lines = [r"\begin{tabular}{l" + "r" * len(labels) + "}", r"\toprule", r"\textbf{Alarm rule} & " + " & ".join(f"\\textbf{{{SHORT[l]}}}" for l in labels) + r" \\", r"\midrule"]
    for name in RULE_STYLE:
        cells = [_pct(_scenario(rules, l)["rules"][name]["rate"]) for l in labels]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER_DIR / "tab_power.tex").write_text("\n".join(lines) + "\n")


def table_reference(reference: dict) -> None:
    lines = [
        r"\begin{tabular}{rrrrrrr}", r"\toprule",
        r"\textbf{Ref.\ rows} & \textbf{Mean (\%)} & \textbf{SD (\%)} & \textbf{$>2\times$ mean} & \textbf{P(alarm$\mid$prev.)} & \textbf{$\geq$1 in 50} & \textbf{if indep.} \\", r"\midrule",
    ]
    for e in reference["by_reference_size"]:
        r = e["rules"]["uncorrected (released)"]
        lines.append(
            f"{e['n_reference']} & {_pct(r['pooled_alarm_rate'], 2)} & {_pct(r['per_reference_rate_sd'], 2)} & {_pct(r['share_of_references_above_2x_pooled'], 0)}\\% & "
            f"{_pct(r['p_alarm_given_previous_alarm'])}\\% & {_pct(r['share_with_alarm_within_50_batches'], 0)}\\% & {_pct(r['share_with_alarm_within_50_if_independent'], 0)}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER_DIR / "tab_reference.tex").write_text("\n".join(lines) + "\n")


def table_autocorr(autocorr: dict) -> None:
    methods = ("naive", "thinned", "ess_adjusted", "block_permutation")
    lines = [
        r"\begin{tabular}{r" + "c" * len(methods) + r"@{\hspace{1.6em}}" + "c" * len(methods) + "}", r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{No drift: flagged (\%)}} & \multicolumn{4}{c}{\textbf{Temperature +0.5 sd: flagged (\%)}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"$\phi$ & Naive & Thin & ESS & Block & Naive & Thin & ESS & Block \\", r"\midrule",
    ]
    for r in autocorr["rows"]:
        cells = [_pct(r["null"][m]["rate"], 2) for m in methods] + [_pct(r["power"]["shift_0.5"][m]["rate"]) for m in methods]
        lines.append(f"{r['phi']:.1f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER_DIR / "tab_autocorr.tex").write_text("\n".join(lines) + "\n")


def table_correlated(correlated: dict) -> None:
    rhos = [r["rho"] for r in correlated["rows"]]
    lines = [
        r"\begin{tabular}{l" + "r" * len(rhos) + r"@{\hspace{1.6em}}" + "r" * len(rhos) + "}", r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{No drift: flagged (\%)}} & \multicolumn{4}{c}{\textbf{All columns +0.25 sd: flagged (\%)}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"\textbf{Alarm rule} & " + " & ".join([f"$\\rho{{=}}{r:.1f}$" for r in rhos] * 2) + r" \\", r"\midrule",
    ]
    for name in RULE_STYLE:
        null = [_pct(r["scenarios"]["no drift"][name]["rate"], 2) for r in correlated["rows"]]
        power = [_pct(r["scenarios"]["all columns +0.25 sd"][name]["rate"]) for r in correlated["rows"]]
        lines.append(f"{name} & " + " & ".join(null + power) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER_DIR / "tab_correlated.tex").write_text("\n".join(lines) + "\n")


def table_power_n(power_n: dict) -> None:
    names = list(power_n["rows"][0]["rules"])
    lines = [r"\begin{tabular}{rr" + "r" * len(names) + "}", r"\toprule", r"\textbf{Batch rows} & \textbf{Ref.\ rows} & " + " & ".join(f"\\textbf{{{n}}}" for n in names) + r" \\", r"\midrule"]
    for row in power_n["rows"]:
        lines.append(f"{row['n_batch']} & {row['n_reference']} & " + " & ".join(_pct(row["rules"][n]["rate"]) for n in names) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER_DIR / "tab_power_n.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    PAPER_DIR.mkdir(exist_ok=True)
    rules, reference, autocorr, power_n = _load("rules.json"), _load("reference.json"), _load("autocorrelation.json"), _load("power_n.json")
    correlated = _load("correlated.json")
    table_correlated(correlated)
    fig_power(rules)
    fig_reference(reference)
    fig_autocorr(autocorr)
    table_null(rules)
    table_power(rules)
    table_reference(reference)
    table_autocorr(autocorr)
    table_power_n(power_n)
    print("figures and tables written to", PAPER_DIR)


if __name__ == "__main__":
    main()
