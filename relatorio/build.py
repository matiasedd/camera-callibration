#!/usr/bin/env python3
"""Builds relatorio/relatorio.html from relatorio.md + figs, for Chrome print-to-pdf."""
import re, markdown

src = open("relatorio/relatorio.md", encoding="utf-8").read()
figno = 0

def single(name, caption, w="88%"):
    global figno
    figno += 1
    return (f'<figure><img style="width:{w}" src="figs/{name}">'
            f'<figcaption>Figura {figno}: {caption}</figcaption></figure>')

def pair(a, b, caption):
    global figno
    figno += 1
    return (f'<figure><div class="pair"><img src="figs/{a}"><img src="figs/{b}"></div>'
            f'<figcaption>Figura {figno}: {caption}</figcaption></figure>')

subs = {
    r"\[FIGURA: cantos\]": lambda: single("cantos.jpg",
        "uma das 20 fotos com os 9x4 cantos internos detectados (script 01)"),
    r"\[FIGURA: matrizes\]": lambda: single("matrizes.png",
        "matriz intrinseca e coeficientes de distorcao obtidos (script 02)", "70%"),
    r"\[FIGURA: undistort\]": lambda: single("undistort.jpg",
        "original x corrigida x mapa de calor da correcao (script 03)", "98%"),
    r"\[FIGURA: retas\]": lambda: pair("retas_antes.jpg", "retas_depois.jpg",
        "fileiras de cantos (colineares no mundo) x corda entre extremos: antes curvam, depois alinham (script 06)"),
    r"\[FIGURA: proj1\]": lambda: single("proj1.jpg",
        "projecao 3D->2D: cantos projetados (amarelo), cubo de 90mm em pe (vermelho), eixos do mundo, ponto flutuante (estrela)"),
    r"\[FIGURA: proj2\]": lambda: single("proj2.jpg",
        "mesmo experimento em outra view — o cubo continua assentado no quadrado"),
    r"\[FIGURA: estereo\]": lambda: single("estereo.png",
        "vista de cima do tabuleiro: posicao real dos cantos x triangulada por 2 views (script 05)"),
}
for pat, fn in subs.items():
    src = re.sub(pat, lambda m, f=fn: f(), src)

body = markdown.markdown(src, extensions=["extra"])

html = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<style>
body {{ font-family: Georgia, serif; max-width: 720px; margin: 40px auto;
       line-height: 1.45; font-size: 14px; color: #111; padding: 0 12px; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 1.6em; }}
h3 {{ font-size: 14px; }}
code {{ font-family: Menlo, monospace; font-size: 12px; background: #f4f4f4; }}
pre code {{ display: block; padding: 10px; }}
a {{ color: #0645ad; }}
figure {{ margin: 18px 0; text-align: center; }}
.pair img {{ width: 49%; }}
figcaption {{ font-size: 12px; color: #555; margin-top: 6px; }}
li {{ margin: 4px 0; }}
</style></head><body>{body}</body></html>"""

open("relatorio/relatorio.html", "w", encoding="utf-8").write(html)
print("relatorio/relatorio.html written,", figno, "figures")
