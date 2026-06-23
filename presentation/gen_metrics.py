#!/usr/bin/env python3
"""Читает experiments/*/metrics.json -> пишет metrics.tex с макросами для слайдов.
Запускается автоматически перед сборкой (см. .latexmkrc) или вручную: python3 gen_metrics.py
Формат чисел тут (.1f) меняет точность во всех слайдах разом."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
EXP = HERE.parent / "experiments"

# каталог (относительно experiments/) -> префикс макроса (только буквы!)
RUNS = {
    "exp_000_trivial/nsk":                     "TrivialNsk",
    "exp_000_trivial/sakhalin":                "TrivialSak",
    "exp_001_baseline/sakhalin_unified":       "Phys",
    "exp_002_transolver":                      "Field",
    "exp_003_transolver_heatmap_multitask":    "Heat",
    "exp_004_transolver_regressor_multitask":  "Reg",
    "exp_005_unet_baseline":                   "Unet",
    "exp_006_transolver_with_wind__no_wind":   "WindNo",
    "exp_006_transolver_with_wind__with_wind": "WindYes",
    "exp_007_pinn":                            "Pinn",
}


def fmt(x: float | None) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    out = ["% автогенерация gen_metrics.py из experiments/*/metrics.json -- руками не править\n"]
    for rel, name in RUNS.items():
        p = EXP / rel / "metrics.json"
        mean = smooth = None
        if p.exists():
            d = json.loads(p.read_text())
            mean, smooth = d.get("mean_error"), d.get("mean_smooth_error")
        out.append(f"\\newcommand{{\\m{name}Mean}}{{{fmt(mean)}}}\n")
        out.append(f"\\newcommand{{\\m{name}Smooth}}{{{fmt(smooth)}}}\n")
    (HERE / "metrics.tex").write_text("".join(out))
    print(f"metrics.tex: {len(RUNS)} методов записано")


if __name__ == "__main__":
    main()
