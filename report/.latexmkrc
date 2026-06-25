# перед каждой сборкой тянем метрики из experiments/*/metrics.json в metrics.tex
# общий скрипт в tools/, пишем metrics.tex рядом (в report/)
system("python3 ../tools/gen_metrics.py metrics.tex");
