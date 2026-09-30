import os
import glob
import cv2 as cv
import numpy as np

# Configurações do tabuleiro
PATTERN_SIZE = (9, 4)        # Cantos internos (largura, altura)
SQUARE_SIZE_MM = 90.0        # Tamanho do lado do quadrado em milímetros
IMG_DIR = "img"
OUTPUT_DIR = os.path.join("Resultados", "img_pontos")
CACHE_FILE = os.path.join("Resultados", "pontos_detectados.npz")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

    if not images:
        print(f"Nenhuma imagem encontrada em '{IMG_DIR}/'.")
        return

    print(f"=== 01: DETECÇÃO DE PONTOS DO TABULEIRO ===")
    print(f"Imagens encontradas: {len(images)}")
    print(f"Tamanho do padrão: {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} cantos internos")
    print(f"Diretório de saída: {OUTPUT_DIR}\n")

    # Coordenadas 3D dos pontos no mundo real (Z = 0)
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []   # Pontos 3D no espaço do mundo real
    imgpoints = []   # Pontos 2D no plano da imagem
    valid_images = []
    img_shape = None

    for idx, fname in enumerate(images, start=1):
        basename = os.path.basename(fname)
        img = cv.imread(fname)

        if img is None:
            print(f"[{idx:02d}/{len(images):02d}] FALHOU : {basename} (Erro ao ler arquivo)")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        # Detector de alta precisão subpixel (Sector Based)
        ret, corners = cv.findChessboardCornersSB(
            gray,
            PATTERN_SIZE,
            flags=cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY
        )

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_images.append(basename)

            # Desenha os cantos detectados na imagem
            img_draw = img.copy()
            cv.drawChessboardCorners(img_draw, PATTERN_SIZE, corners, ret)

            out_path = os.path.join(OUTPUT_DIR, basename)
            cv.imwrite(out_path, img_draw)
            print(f"[{idx:02d}/{len(images):02d}] OK     : {basename} -> Salvo em {out_path}")
        else:
            print(f"[{idx:02d}/{len(images):02d}] FALHOU : {basename}")

    print(f"\nResumo: {len(objpoints)} de {len(images)} imagens aproveitadas com sucesso.")

    # Salva cache dos pontos para acelerar o cálculo das matrizes
    if objpoints:
        np.savez_compressed(
            CACHE_FILE,
            objpoints=np.array(objpoints, dtype=object),
            imgpoints=np.array(imgpoints, dtype=object),
            valid_images=np.array(valid_images),
            img_shape=np.array(img_shape)
        )
        print(f"Cache de pontos salvo em: {CACHE_FILE}")


if __name__ == "__main__":
    main()
