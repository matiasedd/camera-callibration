import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

RESULTADOS_DIR = "Resultados"
CALIB_FILE = os.path.join(RESULTADOS_DIR, "calibracao_camera.npz")
CACHE_POINTS = os.path.join(RESULTADOS_DIR, "pontos_detectados.npz")
OUTPUT_PNG = os.path.join(RESULTADOS_DIR, "estereo_triangulacao.png")

# Duas views com angulos bem diferentes (par "estereo" improvisado)
IMG_A = "20260924_192821.jpg"
IMG_B = "20260924_192852.jpg"
S = 90.0


def main():
    calib = np.load(CALIB_FILE, allow_pickle=True)
    mtx, dist = calib["mtx"], calib["dist"]
    rvecs, tvecs = calib["rvecs"], calib["tvecs"]
    valid_images = [str(x) for x in calib["valid_images"]]

    cache = np.load(CACHE_POINTS, allow_pickle=True)
    objp = np.array(cache["objpoints"][0], dtype=np.float32).reshape(-1, 3)
    imgpoints = [np.array(p, dtype=np.float32).reshape(-1, 2)
                 for p in cache["imgpoints"]]

    ia, ib = valid_images.index(IMG_A), valid_images.index(IMG_B)

    # Cada foto tem pose no referencial do tabuleiro: P = K [R|t]
    def matriz_P(i):
        R, _ = cv.Rodrigues(rvecs[i])
        return mtx @ np.hstack([R, tvecs[i]])

    PA, PB = matriz_P(ia), matriz_P(ib)

    # Remove a distorcao dos pontos 2D antes de triangul ar
    ua = cv.undistortPoints(imgpoints[ia].reshape(-1, 1, 2), mtx, dist,
                            P=mtx).reshape(-1, 2)
    ub = cv.undistortPoints(imgpoints[ib].reshape(-1, 1, 2), mtx, dist,
                            P=mtx).reshape(-1, 2)

    X = cv.triangulatePoints(PA, PB, ua.T, ub.T)
    X = (X[:3] / X[3]).T  # 36x3 no referencial do tabuleiro

    err = np.linalg.norm(X - objp, axis=1)

    print("=== 05: TRIANGULACAO ESTEREO (2 views, mesma camera) ===")
    print(f"Views: {IMG_A}  x  {IMG_B}")
    print(f"Erro de triangulacao: medio {err.mean():.2f} mm  "
          f"max {err.max():.2f} mm  (quadrado = {S:.0f} mm)\n")
    for i in [0, 8, 17, 27, 35]:
        print(f"  canto {i}: real {objp[i]}  "
              f"triangulado {np.round(X[i], 1)}  erro {err[i]:.1f} mm")

    # Figura: vista de cima do tabuleiro, real x triangulado
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(objp[:, 0], objp[:, 1], c="green", s=60, marker="o",
               label="posicao real (tabuleiro)", zorder=3)
    ax.scatter(X[:, 0], X[:, 1], c="red", s=30, marker="x",
               label="triangulado", zorder=4)
    for i in range(len(objp)):
        ax.annotate("", xy=(X[i, 0], X[i, 1]), xytext=(objp[i, 0], objp[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title(f"Triangulacao estereo — erro medio {err.mean():.2f} mm, "
                 f"max {err.max():.2f} mm")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_PNG, dpi=150); plt.close()
    print(f"\nFigura salva em: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
