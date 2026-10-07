# TA02 - Camera Calibration

CI1026 - Visão Computacional e Percepção (UFPR, Prof. Eduardo Todt)

Calibração de uma Samsung Galaxy A24 pelo método de Zhang usando o
tabuleiro planar do VRI (10x5 quadrados de 90 mm, 9x4 cantos internos) —
20 fotos, OpenCV. Remoção de distorção, projeção 3D->2D com conferência e
triangulação "estéreo" entre duas views (bônus).

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy matplotlib
```

## Pipeline (scripts numerados, rodar em ordem)

```sh
python 01_detectar_pontos.py    # cantos 9x4 -> Resultados/img_pontos + cache npz
python 02_obter_matriz.py       # calibrateCamera -> Resultados/calibracao_camera.{npz,json}
python 03_remover_distorcao.py  # original x corrigida x mapa de calor -> img_distorção/
python 04_projetar_pontos.py    # projeta pontos 3D conhecidos nas fotos + confere
python 05_estereo.py            # triangula cantos entre 2 views -> erro em mm
python 06_verificar_retas.py    # fileiras colineares: desvio da reta antes/depois

python relatorio/build.py       # relatorio.md -> relatorio.html (imprimir p/ pdf)
```

## Layout

- `01..06_*.py` — pipeline (cada etapa consome o cache da anterior)
- `notebook-teste.ipynb` — notebook original da exploração no Colab
- `img/` — 20 fotos do tabuleiro (Galaxy A24, 4080x2296)
- `Resultados/` — saídas: calibração (json/npz), figuras, imagens anotadas
- `relatorio/` — relatório (md -> html -> pdf)
