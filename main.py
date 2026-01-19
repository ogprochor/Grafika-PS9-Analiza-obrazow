import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import colorsys


class GreenAreaAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Satellite Image Analyzer - Green Area Detection")
        self.root.geometry("1200x800")

        self.original_image = None
        self.current_image = None
        self.image_path = ""

        # Domyślne parametry do detekcji zieleni
        self.green_params = {
            'hue_min': 30,  # zakres odcieni zieleni (w stopniach 0-360)
            'hue_max': 150,
            'saturation_min': 0.2,  # minimalne nasycenie
            'value_min': 0.15,  # minimalna jasność
            'vegetation_index': True,  # czy używać indeksu roślinności
            'ndvi_threshold': 0.1,  # próg dla NDVI
        }

        # Inne kolory do wykrywania
        self.color_profiles = {
            'Green Areas': {'hue_min': 30, 'hue_max': 150, 'saturation_min': 0.2, 'value_min': 0.15},
            'Water': {'hue_min': 180, 'hue_max': 270, 'saturation_min': 0.3, 'value_min': 0.3},
            'Buildings/Roofs': {'hue_min': 0, 'hue_max': 30, 'saturation_min': 0.1, 'value_min': 0.3},
            'Roads/Concrete': {'hue_min': 0, 'hue_max': 360, 'saturation_min': 0, 'saturation_max': 0.3,
                               'value_min': 0.4},
            'Bare Soil': {'hue_min': 15, 'hue_max': 45, 'saturation_min': 0.2, 'value_min': 0.3},
        }

        self.current_profile = 'Green Areas'
        self.image_label = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lewa część - obraz
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Prawa część - kontrolki
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        self.setup_image_display(left_frame)
        self.setup_controls(right_frame)

    def setup_image_display(self, parent):
        # Górna część - oryginalny obraz
        original_frame = ttk.LabelFrame(parent, text="Original Image", padding=5)
        original_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.canvas_original = tk.Canvas(original_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        self.image_label_original = ttk.Label(self.canvas_original, text="No image loaded")
        self.image_label_original.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Dolna część - maska z wykrytymi obszarami
        mask_frame = ttk.LabelFrame(parent, text="Detected Areas", padding=5)
        mask_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_mask = tk.Canvas(mask_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.canvas_mask.pack(fill=tk.BOTH, expand=True)

        self.image_label_mask = ttk.Label(self.canvas_mask, text="Analysis result will appear here")
        self.image_label_mask.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def setup_controls(self, parent):
        # Przyciski podstawowe
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Reset Image", command=self.reset_image).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(fill=tk.X, pady=2)

        separator = ttk.Separator(parent, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)

        # Wybór profilu kolorów
        ttk.Label(parent, text="Color Profile:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)

        self.profile_var = tk.StringVar(value=self.current_profile)
        profile_combo = ttk.Combobox(parent, textvariable=self.profile_var,
                                     values=list(self.color_profiles.keys()),
                                     state='readonly', width=20)
        profile_combo.pack(fill=tk.X, pady=5)
        profile_combo.bind('<<ComboboxSelected>>', self.on_profile_change)

        separator2 = ttk.Separator(parent, orient='horizontal')
        separator2.pack(fill=tk.X, pady=10)

        # Parametry HSV
        ttk.Label(parent, text="HSV Parameters:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)

        param_frame = ttk.Frame(parent)
        param_frame.pack(fill=tk.X, pady=5)

        # Hue (odcień)
        ttk.Label(param_frame, text="Hue Min:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.hue_min_var = tk.IntVar(value=self.green_params['hue_min'])
        hue_min_scale = ttk.Scale(param_frame, from_=0, to=180, variable=self.hue_min_var,
                                  orient=tk.HORIZONTAL, length=150)
        hue_min_scale.grid(row=0, column=1, padx=5, pady=2)
        self.hue_min_label = ttk.Label(param_frame, text=str(self.green_params['hue_min']))
        self.hue_min_label.grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(param_frame, text="Hue Max:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.hue_max_var = tk.IntVar(value=self.green_params['hue_max'])
        hue_max_scale = ttk.Scale(param_frame, from_=180, to=360, variable=self.hue_max_var,
                                  orient=tk.HORIZONTAL, length=150)
        hue_max_scale.grid(row=1, column=1, padx=5, pady=2)
        self.hue_max_label = ttk.Label(param_frame, text=str(self.green_params['hue_max']))
        self.hue_max_label.grid(row=1, column=2, padx=5, pady=2)

        # Saturation (nasycenie)
        ttk.Label(param_frame, text="Sat Min:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.sat_min_var = tk.DoubleVar(value=self.green_params['saturation_min'])
        sat_min_scale = ttk.Scale(param_frame, from_=0, to=1.0, variable=self.sat_min_var,
                                  orient=tk.HORIZONTAL, length=150)
        sat_min_scale.grid(row=2, column=1, padx=5, pady=2)
        self.sat_min_label = ttk.Label(param_frame, text=f"{self.green_params['saturation_min']:.2f}")
        self.sat_min_label.grid(row=2, column=2, padx=5, pady=2)

        # Value/Lightness (jasność)
        ttk.Label(param_frame, text="Value Min:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.val_min_var = tk.DoubleVar(value=self.green_params['value_min'])
        val_min_scale = ttk.Scale(param_frame, from_=0, to=1.0, variable=self.val_min_var,
                                  orient=tk.HORIZONTAL, length=150)
        val_min_scale.grid(row=3, column=1, padx=5, pady=2)
        self.val_min_label = ttk.Label(param_frame, text=f"{self.green_params['value_min']:.2f}")
        self.val_min_label.grid(row=3, column=2, padx=5, pady=2)

        # Bind updates
        hue_min_scale.configure(command=self.update_hue_min)
        hue_max_scale.configure(command=self.update_hue_max)
        sat_min_scale.configure(command=self.update_sat_min)
        val_min_scale.configure(command=self.update_val_min)

        separator3 = ttk.Separator(parent, orient='horizontal')
        separator3.pack(fill=tk.X, pady=10)

        # Zaawansowane opcje
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Options", padding=10)
        advanced_frame.pack(fill=tk.X, pady=10)

        self.ndvi_var = tk.BooleanVar(value=self.green_params['vegetation_index'])
        ttk.Checkbutton(advanced_frame, text="Use Vegetation Index (NDVI)",
                        variable=self.ndvi_var).pack(anchor=tk.W, pady=2)

        ttk.Label(advanced_frame, text="NDVI Threshold:").pack(anchor=tk.W, pady=2)
        self.ndvi_threshold_var = tk.DoubleVar(value=self.green_params['ndvi_threshold'])
        ndvi_scale = ttk.Scale(advanced_frame, from_=0, to=0.5, variable=self.ndvi_threshold_var,
                               orient=tk.HORIZONTAL)
        ndvi_scale.pack(fill=tk.X, pady=5)
        self.ndvi_label = ttk.Label(advanced_frame, text=f"{self.green_params['ndvi_threshold']:.2f}")
        self.ndvi_label.pack(anchor=tk.W)
        ndvi_scale.configure(command=self.update_ndvi_threshold)

        separator4 = ttk.Separator(parent, orient='horizontal')
        separator4.pack(fill=tk.X, pady=10)

        # Przycisk analizy
        ttk.Button(parent, text="ANALYZE IMAGE", command=self.analyze_image,
                   style='Accent.TButton').pack(fill=tk.X, pady=10)

        # Styl dla przycisku głównego
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'), padding=10)

        # Wyniki
        self.results_frame = ttk.LabelFrame(parent, text="Analysis Results", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.results_text = scrolledtext.ScrolledText(self.results_frame, height=10,
                                                      font=('Courier', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "Load an image and click ANALYZE to see results.")
        self.results_text.configure(state='disabled')

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )

        if file_path:
            try:
                self.image_path = file_path
                self.original_image = np.array(Image.open(file_path))
                self.current_image = self.original_image.copy()

                # Usuń napisy
                if self.image_label_original:
                    self.image_label_original.destroy()
                    self.image_label_original = None
                if self.image_label_mask:
                    self.image_label_mask.destroy()
                    self.image_label_mask = None

                self.display_original_image()
                self.clear_mask_display()

                self.update_results_text("Image loaded successfully. Click ANALYZE to start.")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_original_image()
            self.clear_mask_display()
            self.update_results_text("Image reset to original state.")

    def save_results(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                if file_path.endswith('.txt'):
                    # Zapisz wyniki jako tekst
                    with open(file_path, 'w') as f:
                        f.write(self.results_text.get('1.0', tk.END))
                    messagebox.showinfo("Success", "Results saved as text file")
                else:
                    # Zapisz obraz z maską
                    if hasattr(self, 'mask_image_display'):
                        Image.fromarray(self.mask_image_display).save(file_path)
                        messagebox.showinfo("Success", "Mask image saved successfully")
                    else:
                        messagebox.showwarning("Warning", "No mask image to save")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def display_original_image(self):
        if self.current_image is None:
            return

        pil_image = Image.fromarray(self.current_image)
        self.display_image_on_canvas(pil_image, self.canvas_original)

    def display_mask_image(self, mask_array):
        if mask_array is None:
            return

        # Konwertuj maskę (0-1) na obraz do wyświetlania
        mask_display = (mask_array * 255).astype(np.uint8)

        # Stwórz kolorową wersję (zielone obszary na czarno-białym tle)
        if len(mask_array.shape) == 2:  # Jeśli to maska binarna
            colored_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
            # Zielone obszary
            colored_mask[mask_array == 1] = [0, 255, 0]  # Zielony
            # Tło - oryginalny obraz w odcieniach szarości
            if self.current_image is not None:
                gray = self.rgb_to_grayscale(self.current_image)
                colored_mask[mask_array == 0] = np.stack([gray, gray, gray], axis=-1)[mask_array == 0]

            pil_image = Image.fromarray(colored_mask)
            self.mask_image_display = colored_mask
        else:
            pil_image = Image.fromarray(mask_display)
            self.mask_image_display = mask_display

        self.display_image_on_canvas(pil_image, self.canvas_mask)

    def display_image_on_canvas(self, pil_image, canvas):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            ratio = min(canvas_width / pil_image.width, canvas_height / pil_image.height) * 0.95
            new_width = int(pil_image.width * ratio)
            new_height = int(pil_image.height * ratio)

            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(pil_image)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.tk_image, anchor=tk.CENTER)

    def clear_mask_display(self):
        self.canvas_mask.delete("all")
        self.image_label_mask = ttk.Label(self.canvas_mask, text="Analysis result will appear here")
        self.image_label_mask.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def update_results_text(self, text):
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.configure(state='disabled')

    def on_profile_change(self, event):
        self.current_profile = self.profile_var.get()
        profile = self.color_profiles[self.current_profile]

        # Zaktualizuj parametry
        self.hue_min_var.set(profile['hue_min'])
        self.hue_max_var.set(profile['hue_max'])
        self.sat_min_var.set(profile['saturation_min'])
        self.val_min_var.set(profile['value_min'])

        # Zaktualizuj etykiety
        self.hue_min_label.config(text=str(profile['hue_min']))
        self.hue_max_label.config(text=str(profile['hue_max']))
        self.sat_min_label.config(text=f"{profile['saturation_min']:.2f}")
        self.val_min_label.config(text=f"{profile['value_min']:.2f}")

        self.update_results_text(f"Switched to profile: {self.current_profile}")

    def update_hue_min(self, value):
        self.hue_min_label.config(text=str(int(float(value))))

    def update_hue_max(self, value):
        self.hue_max_label.config(text=str(int(float(value))))

    def update_sat_min(self, value):
        self.sat_min_label.config(text=f"{float(value):.2f}")

    def update_val_min(self, value):
        self.val_min_label.config(text=f"{float(value):.2f}")

    def update_ndvi_threshold(self, value):
        self.ndvi_label.config(text=f"{float(value):.2f}")

    def analyze_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        try:
            # Pobierz aktualne parametry
            params = {
                'hue_min': self.hue_min_var.get(),
                'hue_max': self.hue_max_var.get(),
                'saturation_min': self.sat_min_var.get(),
                'value_min': self.val_min_var.get(),
                'vegetation_index': self.ndvi_var.get(),
                'ndvi_threshold': self.ndvi_threshold_var.get(),
            }

            # Wykonaj analizę
            mask, percentage = self.detect_areas(self.current_image, params)

            # Wyświetl maskę
            self.display_mask_image(mask)

            # Wyświetl wyniki
            results = self.generate_results_text(percentage, params)
            self.update_results_text(results)

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def rgb_to_grayscale(self, img):
        """Konwertuj RGB na skalę szarości"""
        if len(img.shape) == 3:
            return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return img

    def rgb_to_hsv(self, img):
        """Konwertuj RGB na HSV"""
        if len(img.shape) != 3:
            return None

        hsv = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                r, g, b = img[i, j] / 255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv[i, j] = [h * 360, s, v]  # H: 0-360, S: 0-1, V: 0-1

        return hsv

    def calculate_ndvi(self, img):
        """Oblicz Normalized Difference Vegetation Index (NDVI)"""
        if len(img.shape) != 3:
            return None

        # Dla zdjęć satelitarnych: NDVI = (NIR - RED) / (NIR + RED)
        # Dla zwykłych zdjęć RGB: używamy przybliżenia
        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)

        # Przybliżenie NDVI z RGB
        # W roślinności jest dużo chlorofilu, który absorbuje czerwone i odbija bliską podczerwień
        # W zwykłych aparatach NIR jest częściowo rejestrowany przez kanał czerwony
        ndvi = (g - r) / (g + r + 1e-10)  # + mała stała aby uniknąć dzielenia przez 0

        # Normalizuj do zakresu -1 do 1
        ndvi = np.clip(ndvi, -1, 1)

        return ndvi

    def detect_areas(self, img, params):
        """Wykrywanie obszarów na podstawie parametrów"""
        if len(img.shape) != 3:
            raise ValueError("Image must be in color (RGB)")

        height, width = img.shape[:2]
        total_pixels = height * width

        # Metoda 1: Detekcja w przestrzeni HSV
        hsv_img = self.rgb_to_hsv(img)

        # Stwórz maskę HSV
        mask_hsv = np.zeros((height, width), dtype=np.uint8)

        hue_min = params['hue_min']
        hue_max = params['hue_max']
        sat_min = params['saturation_min']
        val_min = params['value_min']

        for i in range(height):
            for j in range(width):
                h, s, v = hsv_img[i, j]
                # Sprawdź czy piksel spełnia warunki
                if (hue_min <= h <= hue_max and
                        s >= sat_min and
                        v >= val_min):
                    mask_hsv[i, j] = 1

        # Metoda 2: NDVI (dla roślinności)
        if params['vegetation_index'] and self.current_profile == 'Green Areas':
            ndvi = self.calculate_ndvi(img)
            if ndvi is not None:
                mask_ndvi = (ndvi > params['ndvi_threshold']).astype(np.uint8)
                # Połącz maski (logiczne OR)
                mask = np.logical_or(mask_hsv, mask_ndvi).astype(np.uint8)
            else:
                mask = mask_hsv
        else:
            mask = mask_hsv

        # Filtracja morfologiczna (opcjonalna) - usuwanie małych obszarów
        mask = self.apply_morphological_filter(mask)

        # Oblicz procent
        green_pixels = np.sum(mask)
        percentage = (green_pixels / total_pixels) * 100

        return mask, percentage

    def apply_morphological_filter(self, mask):
        """Prosta filtracja morfologiczna - usuwanie małych obszarów"""
        # Erozja a potem dylatacja (otwarcie) - usuwa małe obszary
        from scipy import ndimage

        # Erozja
        structure = np.ones((3, 3), dtype=np.uint8)
        eroded = ndimage.binary_erosion(mask, structure=structure)

        # Dylatacja
        dilated = ndimage.binary_dilation(eroded, structure=structure)

        return dilated.astype(np.uint8)

    def generate_results_text(self, percentage, params):
        """Generuj tekst z wynikami"""
        text = f"=== ANALYSIS RESULTS ===\n\n"
        text += f"Color Profile: {self.current_profile}\n"
        text += f"Total Pixels: {self.current_image.shape[0] * self.current_image.shape[1]:,}\n"
        text += f"Detected Area: {percentage:.2f}%\n\n"

        text += f"Parameters Used:\n"
        text += f"  Hue Range: {params['hue_min']}° - {params['hue_max']}°\n"
        text += f"  Min Saturation: {params['saturation_min']:.2f}\n"
        text += f"  Min Value: {params['value_min']:.2f}\n"
        text += f"  Vegetation Index: {'Yes' if params['vegetation_index'] else 'No'}\n"
        if params['vegetation_index']:
            text += f"  NDVI Threshold: {params['ndvi_threshold']:.2f}\n"

        text += f"\nInterpretation:\n"
        if self.current_profile == 'Green Areas':
            if percentage < 10:
                text += "  Very low green area coverage (urban area)\n"
            elif percentage < 25:
                text += "  Low green area coverage\n"
            elif percentage < 40:
                text += "  Moderate green area coverage\n"
            elif percentage < 60:
                text += "  High green area coverage\n"
            else:
                text += "  Very high green area coverage (park/forest)\n"

        text += f"\nAnalysis completed successfully.\n"

        return text


if __name__ == "__main__":
    root = tk.Tk()
    app = GreenAreaAnalyzer(root)
    root.mainloop()