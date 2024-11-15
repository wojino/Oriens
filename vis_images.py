import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import os

start = 8
end = 213
image_pairs = [(f"{i:03d}.png", f"{i:03d}_m.png") for i in range(start, end)]
image_dir = "experiments/plots"

sets_per_page = 12
fig_width, fig_height = 60, 120
dpi = 100
gs = GridSpec(sets_per_page, 2, width_ratios=[5, 1])

plt.figure(figsize=(60, 120), dpi=100)

gs = GridSpec(len(image_pairs), 2, width_ratios=[5, 1])

with PdfPages("experiments/plots/all_pages.pdf") as pdf:
    for page_start in range(0, len(image_pairs), sets_per_page):
        # 새로운 페이지 시작
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        # GridSpec을 사용하여 12세트 배치 (세트마다 2개의 열)
        gs = GridSpec(sets_per_page, 2, width_ratios=[5, 1])

        # 페이지의 이미지 세트 로드
        for idx, (img_file, img_m_file) in enumerate(
            image_pairs[page_start : page_start + sets_per_page]
        ):
            img_path = os.path.join(image_dir, img_file)
            img_m_path = os.path.join(image_dir, img_m_file)

            img = mpimg.imread(img_path)
            img_m = mpimg.imread(img_m_path)

            plt.subplot(gs[idx, 0])
            plt.imshow(img)
            plt.axis("off")

            plt.subplot(gs[idx, 1])
            plt.imshow(img_m)
            plt.axis("off")

        # 여백을 최소화하여 레이아웃 조정
        plt.tight_layout()

        # 현재 페이지를 PDF에 저장
        pdf.savefig()
        plt.close()
