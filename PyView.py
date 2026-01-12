import customtkinter as ctk
from PIL import Image
import os
import datetime

# --- 設定 ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class ModernImageGridApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Captures Gallery")
        self.geometry("1000x800")

        # 監視対象設定
        # このスクリプトのディレクトリを取得
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.target_folder = os.path.join(script_dir, "captures")

        self.last_files_set = set()
        self.image_refs = []
        self.card_frames = []
        self._resize_job = None

        # --- UI構築 ---

        # 1. ヘッダーエリア
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=20, pady=(20, 10))

        self.lbl_title = ctk.CTkLabel(self.header_frame, text="Captures Gallery", font=("Roboto Medium", 24))
        self.lbl_title.pack(side="left")

        self.lbl_status = ctk.CTkLabel(self.header_frame, text="読み込み中...", text_color="gray")
        self.lbl_status.pack(side="right", padx=10)

        self.btn_reload = ctk.CTkButton(self.header_frame, text="⟳ 更新", width=80, command=self.force_reload)
        self.btn_reload.pack(side="right")

        # 2. メインスクロールエリア
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ウィンドウサイズ変更イベント
        self.bind("<Configure>", self.on_resize)

        # --- 設定値 ---
        self.CARD_WIDTH = 220
        self.IMG_SIZE = (200, 200)

        # 初回実行
        self.check_and_update()
        self.after(1000, self.monitor_folder)

    def show_large_image(self, file_path):
        """ 画像拡大用のモダンな別ウィンドウ """
        try:
            top = ctk.CTkToplevel(self)
            top.title(os.path.basename(file_path))
            top.geometry("900x900")
            top.attributes("-topmost", True)
            top.focus()

            pil_img = Image.open(file_path)
            w, h = pil_img.size
            max_size = 800
            scale = min(max_size/w, max_size/h)
            new_size = (int(w*scale), int(h*scale))

            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=new_size)

            lbl = ctk.CTkLabel(top, text="", image=ctk_img)
            lbl.pack(expand=True, fill="both", padx=20, pady=20)

            top.protocol("WM_DELETE_WINDOW", top.destroy)

        except Exception as e:
            print(f"Error opening large image: {e}")

    def monitor_folder(self):
        if not os.path.exists(self.target_folder):
            self.after(1000, self.monitor_folder)
            return

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        try:
            current_files = {f for f in os.listdir(self.target_folder) if os.path.splitext(f)[1].lower() in valid_extensions}
        except:
            current_files = set()

        if current_files != self.last_files_set:
            self.lbl_status.configure(text="更新中...", text_color="#3B8ED0")
            self.after(10, self.check_and_update)

        self.after(1000, self.monitor_folder)

    def force_reload(self):
        self.lbl_status.configure(text="更新中...", text_color="#3B8ED0")
        self.after(10, self.check_and_update)

    def check_and_update(self):
        self.load_images()
        if os.path.exists(self.target_folder):
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            self.last_files_set = {f for f in os.listdir(self.target_folder) if os.path.splitext(f)[1].lower() in valid_extensions}

        now = datetime.datetime.now()
        update_time_str = f"最終更新: {now.strftime('%H:%M:%S')}"
        self.lbl_status.configure(text=update_time_str, text_color="#2CC985")

    def load_images(self):
        if not os.path.exists(self.target_folder):
            try: os.makedirs(self.target_folder)
            except: return

        # UIクリア時はグリッド設定も忘れる
        for widget in self.scroll_frame.winfo_children():
            widget.grid_forget()
            widget.destroy()
        self.image_refs.clear()
        self.card_frames.clear()

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        try:
            files = [f for f in os.listdir(self.target_folder) if os.path.splitext(f)[1].lower() in valid_extensions]
        except: return

        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.target_folder, x)), reverse=True)

        if not files:
            lbl = ctk.CTkLabel(self.scroll_frame, text="画像がありません", font=("Meiryo", 16))
            lbl.pack(pady=50)
            return

        for filename in files:
            file_path = os.path.join(self.target_folder, filename)
            try:
                pil_img = Image.open(file_path)
                img_w, img_h = pil_img.size
                aspect = img_w / img_h
                if aspect > 1:
                    disp_w = self.IMG_SIZE[0]
                    disp_h = int(self.IMG_SIZE[0] / aspect)
                else:
                    disp_h = self.IMG_SIZE[1]
                    disp_w = int(self.IMG_SIZE[1] * aspect)

                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))
                self.image_refs.append(ctk_img)

                card = ctk.CTkFrame(self.scroll_frame, width=self.CARD_WIDTH, fg_color=("gray85", "gray20"), corner_radius=10)

                lbl_img = ctk.CTkLabel(card, text="", image=ctk_img)
                lbl_img.pack(padx=10, pady=(10, 5))

                lbl_name = ctk.CTkLabel(card, text=filename, font=("Meiryo", 10), wraplength=180)
                lbl_name.pack(padx=10, pady=(0, 10))

                handler = lambda e, path=file_path: self.show_large_image(path)
                lbl_img.bind("<Double-Button-1>", handler)
                lbl_name.bind("<Double-Button-1>", handler)
                card.bind("<Double-Button-1>", handler)

                card.bind("<Enter>", lambda e, c=card: c.configure(border_width=1, border_color="#3B8ED0"))
                card.bind("<Leave>", lambda e, c=card: c.configure(border_width=0))

                self.card_frames.append(card)

            except Exception as e:
                print(f"Error loading {filename}: {e}")

        # ★修正ポイント1: 画像配置後、少し待ってからグリッド計算を行う
        # これによりスクロールバーの有無が確定した後に正しく計算される
        self.after(50, self.rearrange_grid)

    def on_resize(self, event):
        # メインウィンドウのリサイズ時のみ再計算
        if event.widget == self:
            # 既存のタイマーをキャンセル
            if self._resize_job:
                self.after_cancel(self._resize_job)
            # 250ms後に再計算を実行する新しいタイマーをセット
            self._resize_job = self.after(250, self.rearrange_grid)

    def rearrange_grid(self):
        if not self.card_frames:
            return

        # スクロールフレームの現在の幅を取得
        width = self.scroll_frame.winfo_width()

        # ★修正ポイント3: 幅がまだ正しく取れない場合は再試行するガード処理
        if width < self.CARD_WIDTH:
            self.after(50, self.rearrange_grid)
            return

        # 列数計算 (カード幅 + マージン分を考慮)
        item_full_width = self.CARD_WIDTH + 20
        columns = max(1, int(width / item_full_width))

        # グリッド配置を一旦クリアしてから再配置
        for card in self.card_frames:
            card.grid_forget()

        for i, card in enumerate(self.card_frames):
            row = i // columns
            col = i % columns
            card.grid(row=row, column=col, padx=10, pady=10)

if __name__ == "__main__":
    app = ModernImageGridApp()
    app.mainloop()