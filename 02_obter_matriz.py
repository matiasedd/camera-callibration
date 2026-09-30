import os
import glob
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

PATTERN_SIZE = (9, 4)
SQUARE_SIZE_MM = 90.0
IMG_DIR = "img"
RESULTADOS_DIR = "Resultados"
CACHE_POINTS = os.path.join(RESULTADOS_DIR, "pontos_detectados.npz")
OUTPUT_NPZ = os.path.join(RESULTADOS_DIR, "calibracao_camera.npz")
OUTPUT_JSON = os.path.join(RESULTADOS_DIR, "calibracao_camera.json")
OUTPUT_PNG = os.path.join(RESULTADOS_DIR, "grafico_matrizes.png")


def carregar_ou_detectar_pontos():
    """Carrega os pontos do cache ou realiza a detecção se o cache não existir."""
    if os.path.exists(CACHE_POINTS):
        print(f"Carregando pontos previamente detectados de: {CACHE_POINTS}")
        data = np.load(CACHE_POINTS, allow_pickle=True)
        objpoints = [np.array(p, dtype=np.float32) for p in data["objpoints"]]
        imgpoints = [np.array(p, dtype=np.float32) for p in data["imgpoints"]]
        valid_images = [str(x) for x in data["valid_images"]]
        img_shape = tuple(int(x) for x in data["img_shape"])
        return objpoints, imgpoints, valid_images, img_shape

    print("Cache não encontrado. Detectando cantos nas imagens...")
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints, imgpoints, valid_images = [], [], []
    img_shape = None

    for fname in images:
        basename = os.path.basename(fname)
        img = cv.imread(fname)
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret, corners = cv.findChessboardCornersSB(
            gray, PATTERN_SIZE, flags=cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY
        )
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_images.append(basename)

    return objpoints, imgpoints, valid_images, img_shape


def main():
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    objpoints, imgpoints, valid_images, img_shape = carregar_ou_detectar_pontos()

    if not objpoints:
        print("Nenhum ponto válido encontrado para calibração.")
        return

    print("\nExecutando calibração da câmera com cv.calibrateCamera...")
    ret_rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()[:5]

    # Cálculo dos erros de reprojeção por imagem e distâncias Z
    erros_por_imagem = []
    distancias_z = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        p_orig = imgpoints[i].reshape(-1, 2)
        p_proj = imgpoints2.reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(p_orig - p_proj, axis=1)))
        z_mm = float(tvecs[i][2, 0])
        erros_por_imagem.append((valid_images[i], err, z_mm))
        distancias_z.append(z_mm)

    print(f"\nCalibração concluída com sucesso!")
    print(f"-> Erro RMS Global: {ret_rms:.4f} pixels ({img_shape[0]}x{img_shape[1]} px)")

    # Salva em formato NPZ (NumPy)
    np.savez_compressed(
        OUTPUT_NPZ,
        mtx=mtx,
        dist=dist,
        rms=ret_rms,
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        img_shape=np.array(img_shape),
        valid_images=np.array(valid_images)
    )

    # Salva em formato JSON
    dados_json = {
        "rms_global": float(ret_rms),
        "resolucao": {"largura": int(img_shape[0]), "altura": int(img_shape[1])},
        "matriz_intrinseca": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "matriz": mtx.tolist()
        },
        "coeficientes_distorcao": {
            "k1": float(k1),
            "k2": float(k2),
            "p1": float(p1),
            "p2": float(p2),
            "k3": float(k3),
            "vetor": dist.ravel().tolist()
        },
        "parametros_extrinsecos_por_imagem": [
            {
                "imagem": name,
                "erro_reprojecao_px": float(err),
                "distancia_z_mm": float(z_mm),
                "rvec": rvecs[i].ravel().tolist(),
                "tvec": tvecs[i].ravel().tolist()
            }
            for i, (name, err, z_mm) in enumerate(erros_por_imagem)
        ]
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dados_json, f, indent=2, ensure_ascii=False)

    # Gera gráfico visual das tabelas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.2, 1.8]})
    ax1.axis("off")
    ax1.set_title("Matriz da Câmera (mtx)\n[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]", fontsize=11, pad=10)
    cell_mtx = [[f"{val:.2f}" for val in row] for row in mtx]
    t1 = ax1.table(cellText=cell_mtx, loc="center", cellLoc="center", rowLabels=["X", "Y", "Z"], colLabels=["X", "Y", "Z"])
    t1.scale(1, 1.8)

    ax2.axis("off")
    ax2.set_title("Coeficientes de Distorção (dist)\n[k1, k2, p1, p2, k3]", fontsize=11, pad=10)
    cell_dist = [[f"{val:.4f}" for val in dist.ravel()[:5]]]
    t2 = ax2.table(cellText=cell_dist, loc="center", cellLoc="center", colLabels=["k1", "k2", "p1", "p2", "k3"])
    t2.scale(1, 1.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()

    print("\nArquivos gerados:")
    print(f"  -> {OUTPUT_NPZ}  (binário NumPy para uso no script 03)")
    print(f"  -> {OUTPUT_JSON} (dados completos em JSON)")
    print(f"  -> {OUTPUT_PNG}  (gráfico das matrizes)")


if __name__ == "__main__":
    main()
