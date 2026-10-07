import os
import glob
import cv2 as cv
import numpy as np

IMG_DIR = "img"
RESULTADOS_DIR = "Resultados"
CALIB_FILE = os.path.join(RESULTADOS_DIR, "calibracao_camera.npz")
CACHE_POINTS = os.path.join(RESULTADOS_DIR, "pontos_detectados.npz")
OUTPUT_DIR = os.path.join(RESULTADOS_DIR, "img_projecao")

# Fotos escolhidas para o relatorio (angulos variados)
PICKS = [
    "20260924_192821.jpg", "20260924_192828.jpg",
    "20260924_192841.jpg", "20260924_192852.jpg",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    calib = np.load(CALIB_FILE, allow_pickle=True)
    mtx, dist = calib["mtx"], calib["dist"]
    rvecs, tvecs = calib["rvecs"], calib["tvecs"]
    valid_images = [str(x) for x in calib["valid_images"]]

    cache = np.load(CACHE_POINTS, allow_pickle=True)
    objpoints = [np.array(p, dtype=np.float32) for p in cache["objpoints"]]
    imgpoints = [np.array(p, dtype=np.float32) for p in cache["imgpoints"]]

    S = 90.0  # lado do quadrado em mm
    # Cubo de lado S "em pe" sobre o quadrado do canto (4,1) do grid
    # (Z negativo = para fora do plano do tabuleiro, na direcao da camera)
    bx, by = 4 * S, 1 * S
    cubo = np.array([
        [bx, by, 0], [bx + S, by, 0], [bx + S, by + S, 0], [bx, by + S, 0],
        [bx, by, -S], [bx + S, by, -S], [bx + S, by + S, -S], [bx, by + S, -S],
    ], np.float32)
    arestas = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
               (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    # Eixos do mundo na origem (270 mm) e um ponto flutuante 180 mm acima
    eixos = np.array([[0, 0, 0], [3 * S, 0, 0], [0, 3 * S, 0], [0, 0, -3 * S]],
                     np.float32)
    hover = np.array([[4 * S, 1.5 * S, -2 * S]], np.float32)

    print("=== 04: PROJECAO 3D -> IMAGEM ===")
    print("Referencial: plano do tabuleiro = Z 0, origem no canto 0, mm\n")

    for i, name in enumerate(valid_images):
        rvec, tvec = rvecs[i], tvecs[i]

        # Conferencia numerica: projeta cantos conhecidos x detectados
        proj, _ = cv.projectPoints(objpoints[i], rvec, tvec, mtx, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - imgpoints[i].reshape(-1, 2),
                             axis=1)
        print(f"{name}: reprojecao media {err.mean():.2f} px  max {err.max():.2f} px")

        if name not in PICKS:
            continue

        img = cv.imread(os.path.join(IMG_DIR, name))

        # Cantos projetados (amarelo) ligados aos detectados (magenta)
        for p, d in zip(proj.reshape(-1, 2),
                        imgpoints[i].reshape(-1, 2)):
            cv.line(img, tuple(p.astype(int)), tuple(d.astype(int)),
                    (255, 0, 255), 3)
            cv.circle(img, tuple(p.astype(int)), 8, (0, 255, 255), 3)

        P, _ = cv.projectPoints(cubo, rvec, tvec, mtx, dist)
        P = P.reshape(-1, 2).astype(int)
        for a, b in arestas:
            cv.line(img, tuple(P[a]), tuple(P[b]), (0, 0, 255), 4)

        A, _ = cv.projectPoints(eixos, rvec, tvec, mtx, dist)
        A = A.reshape(-1, 2).astype(int)
        for k, cor in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv.arrowedLine(img, tuple(A[0]), tuple(A[k + 1]), cor, 4,
                           tipLength=0.15)

        H, _ = cv.projectPoints(hover, rvec, tvec, mtx, dist)
        cv.drawMarker(img, tuple(H.reshape(-1, 2).astype(int)[0]),
                      (255, 255, 0), cv.MARKER_STAR, 40, 4)

        out = os.path.join(OUTPUT_DIR, name)
        cv.imwrite(out, img)
        print(f"  -> anotada: {out}")

    print("\nLegenda: amarelo=canto projetado, magenta=proj->detectado,")
    print("vermelho=cubo 3D, RGB=eixos XYZ, estrela=ponto flutuante 180mm")


if __name__ == "__main__":
    main()
