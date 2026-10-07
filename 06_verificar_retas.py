import os
import cv2 as cv
import numpy as np

IMG_DIR = "img"
RESULTADOS_DIR = "Resultados"
CALIB_FILE = os.path.join(RESULTADOS_DIR, "calibracao_camera.npz")
CACHE_POINTS = os.path.join(RESULTADOS_DIR, "pontos_detectados.npz")
OUTPUT_DIR = os.path.join(RESULTADOS_DIR, "img_retas")

IMG = "20260924_192821.jpg"
PATTERN = (9, 4)  # 4 fileiras de 9 cantos


def desvio_da_corda(pts):
    """Desvio max/medio dos pontos intermediarios em relacao a reta entre as pontas."""
    a, b = pts[0], pts[-1]
    v = (b - a) / np.linalg.norm(b - a)
    n = np.array([-v[1], v[0]])
    devs = [abs(np.dot(q - a, n)) for q in pts[1:-1]]
    return max(devs), float(np.mean(devs))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    calib = np.load(CALIB_FILE, allow_pickle=True)
    mtx, dist = calib["mtx"], calib["dist"]
    img_shape = tuple(int(x) for x in calib["img_shape"])
    valid = [str(x) for x in calib["valid_images"]]

    cache = np.load(CACHE_POINTS, allow_pickle=True)
    imgpoints = [np.array(p, dtype=np.float32).reshape(-1, 2)
                 for p in cache["imgpoints"]]
    det = imgpoints[valid.index(IMG)]

    img = cv.imread(os.path.join(IMG_DIR, IMG))
    h, w = img.shape[:2]

    # Uma fileira de cantos e COLINEAR no mundo -> na imagem ideal vira reta.
    # Medimos quanto os cantos detectados desviam da corda entre os extremos,
    # antes e depois de remover a distorcao.
    mtx2, _ = cv.getOptimalNewCameraMatrix(mtx, dist, img_shape, 0.8, img_shape)
    und = cv.undistort(img, mtx, dist, None, mtx2)
    det_u = cv.undistortPoints(det.reshape(-1, 1, 2), mtx, dist,
                             P=mtx2).reshape(-1, 2)

    print("=== 06: VERIFICACAO DE RETAS (fileiras de cantos) ===\n")
    for label, im, pts in [("ANTES (original)", img, det),
                           ("DEPOIS (corrigida)", und, det_u)]:
        out = im.copy()
        print(label)
        for row in range(PATTERN[1]):
            p = pts[row * PATTERN[0]:(row + 1) * PATTERN[0]]
            mx, mn = desvio_da_corda(p)
            cv.line(out, tuple(p[0].astype(int)), tuple(p[-1].astype(int)),
                    (0, 0, 255), 3)
            for q in p:
                cv.circle(out, tuple(q.astype(int)), 10, (0, 255, 255), -1)
            print(f"  fileira {row}: desvio max {mx:.2f} px  medio {mn:.2f} px")
        tag = "antes" if "ANTES" in label else "depois"
        cv.imwrite(os.path.join(OUTPUT_DIR, f"retas_{tag}.jpg"), out)
    print(f"\nImagens salvas em: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
