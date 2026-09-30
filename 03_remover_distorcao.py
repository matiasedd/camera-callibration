import os
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = "img"
RESULTADOS_DIR = "Resultados"
CALIB_FILE = os.path.join(RESULTADOS_DIR, "calibracao_camera.npz")
OUTPUT_DIR = os.path.join(RESULTADOS_DIR, "img_distorção")


def obter_calibracao():
    """Carrega os parâmetros intrínsecos e de distorção ou executa a calibração se necessário."""
    if os.path.exists(CALIB_FILE):
        print(f"Carregando calibração de: {CALIB_FILE}")
        data = np.load(CALIB_FILE, allow_pickle=True)
        return data["mtx"], data["dist"], tuple(data["img_shape"])

    print("Arquivo de calibração não encontrado. Execute '02_obter_matriz.py' primeiro.")
    raise FileNotFoundError(f"Calibração não encontrada em {CALIB_FILE}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mtx, dist, img_shape = obter_calibracao()

    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not images:
        print(f"Nenhuma imagem encontrada em '{IMG_DIR}/'.")
        return

    print(f"\n=== 03: REMOÇÃO DE DISTORÇÃO E GERAÇÃO DAS COMPARAÇÕES ===")
    print(f"Total de imagens a processar: {len(images)}")
    print(f"Diretório de saída: {OUTPUT_DIR}\n")

    for idx, fname in enumerate(images, start=1):
        basename = os.path.basename(fname)
        name_no_ext, _ = os.path.splitext(basename)

        img_bgr = cv.imread(fname)
        if img_bgr is None:
            print(f"[{idx:02d}/{len(images):02d}] FALHOU : {basename} (Erro ao ler arquivo)")
            continue

        h, w = img_bgr.shape[:2]

        # Otimiza a matriz intrínseca para manter enquadramento adequado
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Aplica a remoção da distorção óptica da lente
        undist_bgr = cv.undistort(img_bgr, mtx, dist, None, newcameramtx)

        # Conversão para RGB para exibição correta no Matplotlib
        orig_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        undist_rgb = cv.cvtColor(undist_bgr, cv.COLOR_BGR2RGB)

        # Cálculo da diferença absoluta de cor (mapa de calor)
        diff = cv.absdiff(orig_rgb, undist_rgb)
        diff_gray = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)

        # Criação da figura composta com 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(orig_rgb)
        axes[0].set_title(f"1. Original (Com Distorção)\n{basename}", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(undist_rgb)
        axes[1].set_title("2. Corrigida (Sem Distorção)", fontsize=11, color="green")
        axes[1].axis("off")

        im3 = axes[2].imshow(diff_gray, cmap="inferno")
        axes[2].set_title("3. Mapa de Calor da Correção\n|Original - Corrigida|", fontsize=11, color="darkorange")
        axes[2].axis("off")

        cbar = fig.colorbar(im3, ax=axes[2], fraction=0.035, pad=0.04)
        cbar.set_label("Intensidade da Correção", rotation=270, labelpad=15)

        plt.tight_layout()

        out_fname = f"{name_no_ext}_comparacao.png"
        out_path = os.path.join(OUTPUT_DIR, out_fname)
        plt.savefig(out_path, dpi=130)
        plt.close(fig)

        print(f"[{idx:02d}/{len(images):02d}] OK : {basename} -> Salvo em {out_path}")

    print(f"\nConcluído! Todas as {len(images)} imagens foram corrigidas e salvas em '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
