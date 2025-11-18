import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

from tkinter import (
    Tk, Frame, Label, Button, filedialog, messagebox, BOTH, LEFT, RIGHT, TOP, BOTTOM, X
)
from PIL import Image, ImageTk

import numpy as np
import matplotlib.pyplot as plt

from gradcam_utils import GradCAM, load_image_as_tensor, overlay_cam_on_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/densenet121_pneumonia.pth"
DATA_DIR = "data/chest_xray"          # корень датасета Kaggle
TEST_DIR = os.path.join(DATA_DIR, "test")


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    classes = checkpoint["classes"]

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(DEVICE)
    model.eval()

    return model, classes


def predict_image(model, img_tensor):
    img_tensor = img_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item(), probs.cpu().numpy()[0]


class PneumoniaApp(Tk):
    def __init__(self):
        super().__init__()

        # --- базовые настройки окна ---
        self.title("Pneumonia Detection - DenseNet121 + Grad-CAM")
        self.geometry("1100x700")
        self.configure(bg="#1e1e1e")  # тёмный фон

        # модели
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Ошибка",
                f"Файл модели не найден: {MODEL_PATH}\nСначала запусти train_densenet.py."
            )
            self.destroy()
            return

        self.model, self.classes = load_model(MODEL_PATH)
        self.gradcam = GradCAM(self.model)

        # для отображения
        self.image_path = None
        self.original_pil = None
        self.original_tk = None
        self.gradcam_tk = None

        # трансформация для test-набора (для confusion matrix)
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # --- layout ---
        self._build_layout()

    def _build_layout(self):
        # верхняя панель с кнопками
        top_frame = Frame(self, bg="#252526")
        top_frame.pack(side=TOP, fill=X, padx=10, pady=10)

        btn_load = Button(
            top_frame,
            text="📂 Загрузить снимок",
            command=self.load_image,
            bg="#0e639c",
            fg="white",
            activebackground="#1177bb",
            relief="flat",
            padx=15,
            pady=6
        )
        btn_load.pack(side=LEFT, padx=5)

        self.btn_predict = Button(
            top_frame,
            text="🩺 Предсказать",
            command=self.run_prediction,
            bg="#3c873a",
            fg="white",
            activebackground="#46a049",
            relief="flat",
            padx=15,
            pady=6,
            state="disabled"
        )
        self.btn_predict.pack(side=LEFT, padx=5)

        btn_cm = Button(
            top_frame,
            text="📊 Confusion Matrix (test)",
            command=self.show_confusion_matrix,
            bg="#6439b7",
            fg="white",
            activebackground="#7d4ede",
            relief="flat",
            padx=15,
            pady=6,
        )
        btn_cm.pack(side=LEFT, padx=5)

        # центральная часть: изображения
        center_frame = Frame(self, bg="#1e1e1e")
        center_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=(0, 10))

        # левое изображение (оригинал)
        left_img_frame = Frame(center_frame, bg="#1e1e1e")
        left_img_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        Label(
            left_img_frame,
            text="Оригинальный снимок",
            bg="#1e1e1e",
            fg="white",
            font=("Helvetica", 12, "bold")
        ).pack(side=TOP, pady=5)

        self.original_label = Label(left_img_frame, bg="#252526")
        self.original_label.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        # правое изображение (Grad-CAM)
        right_img_frame = Frame(center_frame, bg="#1e1e1e")
        right_img_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=5)

        Label(
            right_img_frame,
            text="Grad-CAM Heatmap",
            bg="#1e1e1e",
            fg="white",
            font=("Helvetica", 12, "bold")
        ).pack(side=TOP, pady=5)

        self.gradcam_label = Label(right_img_frame, bg="#252526")
        self.gradcam_label.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        # нижняя панель с результатами
        bottom_frame = Frame(self, bg="#252526")
        bottom_frame.pack(side=BOTTOM, fill=X, padx=10, pady=10)

        self.pred_label = Label(
            bottom_frame,
            text="Класс: —",
            bg="#252526",
            fg="white",
            font=("Helvetica", 14, "bold")
        )
        self.pred_label.pack(side=TOP, anchor="w", pady=2)

        self.conf_label = Label(
            bottom_frame,
            text="Уверенность: —",
            bg="#252526",
            fg="white",
            font=("Helvetica", 12)
        )
        self.conf_label.pack(side=TOP, anchor="w", pady=2)

        self.prob_label = Label(
            bottom_frame,
            text="Вероятности по классам: —",
            bg="#252526",
            fg="white",
            font=("Helvetica", 12),
            justify="left"
        )
        self.prob_label.pack(side=TOP, anchor="w", pady=2)

    # ---------- работа с изображениями ----------

    def _resize_for_display(self, pil_img, max_size=(500, 500)):
        img = pil_img.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите рентген-снимок лёгких",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if not path:
            return

        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return

        self.image_path = path
        self.original_pil = pil_img

        # отображаем оригинал
        disp_img = self._resize_for_display(pil_img)
        self.original_tk = ImageTk.PhotoImage(disp_img)
        self.original_label.configure(image=self.original_tk)

        # очищаем Grad-CAM и текст
        self.gradcam_label.configure(image="")
        self.gradcam_tk = None
        self.pred_label.configure(text="Класс: —", fg="white")
        self.conf_label.configure(text="Уверенность: —")
        self.prob_label.configure(text="Вероятности по классам: —")

        # включаем кнопку предсказания
        self.btn_predict.configure(state="normal")

    def run_prediction(self):
        if self.image_path is None:
            messagebox.showwarning("Внимание", "Сначала загрузите снимок.")
            return

        pil_img, img_tensor = load_image_as_tensor(self.image_path)

        pred_idx, conf, probs = predict_image(self.model, img_tensor)
        pred_class = self.classes[pred_idx]
        conf_percent = conf * 100

        # цвет текста в зависимости от диагноза
        if pred_class.lower() == "pneumonia":
            color = "#ff5555"  # красный
        else:
            color = "#4ec9b0"  # зелёный

        self.pred_label.configure(text=f"Класс: {pred_class}", fg=color)
        self.conf_label.configure(text=f"Уверенность: {conf_percent:.2f}%")

        probs_text_lines = []
        for cl, p in zip(self.classes, probs):
            probs_text_lines.append(f"  {cl}: {p * 100:.2f}%")
        probs_text = "Вероятности по классам:\n" + "\n".join(probs_text_lines)

        self.prob_label.configure(text=probs_text)

        # --- Grad-CAM ---
        cam = self.gradcam.generate(img_tensor, target_class=pred_idx)
        overlay_np = overlay_cam_on_image(self.original_pil, cam)

        overlay_pil = Image.fromarray(overlay_np)
        overlay_disp = self._resize_for_display(overlay_pil)
        self.gradcam_tk = ImageTk.PhotoImage(overlay_disp)
        self.gradcam_label.configure(image=self.gradcam_tk)

    # ---------- Confusion Matrix ----------

    def show_confusion_matrix(self):
        if not os.path.isdir(TEST_DIR):
            messagebox.showerror(
                "Ошибка",
                f"Папка с тестовыми данными не найдена:\n{TEST_DIR}"
            )
            return

        # загружаем test-набор
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=self.test_transform)

        # проверим, что порядок классов совпадает
        if test_dataset.classes != list(self.classes):
            # если хотят, можно вывести предупреждение
            print("Внимание: порядок классов в test-наборе и в модели отличается.")
            print("Модель:", self.classes)
            print("Test dataset:", test_dataset.classes)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2
        )

        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        num_classes = len(self.classes)
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for t, p in zip(all_labels, all_preds):
            cm[t, p] += 1

        # визуализация матрицы
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=self.classes,
            yticklabels=self.classes,
            ylabel='Истинный класс',
            xlabel='Предсказанный класс',
            title='Confusion Matrix (Test set)'
        )

        # подписи в ячейках
        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        fig.tight_layout()
        plt.show()


def main():
    app = PneumoniaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
