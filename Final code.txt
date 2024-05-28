import torch
from PIL import Image, ImageTk
from torchvision.transforms import ToTensor, ToPILImage
from RRDBNet_arch import RRDBNet
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time


def load_model(model_path):
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model


def upscale_image(model, img, scale_factor, progress_bar):
    
    img_tensor = ToTensor()(img).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    
    with torch.no_grad():
        output = model(img_tensor)

    
    output_img = output.squeeze().clamp(0, 1).cpu()
    output_img = ToPILImage()(output_img)

    if progress_bar is not None:
        progress_bar["value"] = 100
        progress_bar.update()

    return output_img


def estimate_processing_time(model, img, scale_factor):
    
    img_crop = img.crop((0, 0, img.width // 10, img.height // 10))
    start_time = time.time()
    
    
    upscale_image(model, img_crop, scale_factor, None)
    
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    
    estimated_time = processing_time * 100  
    estimated_time_minutes = estimated_time / 60  
    return estimated_time_minutes


model_path = 'RRDB_ESRGAN_x4.pth'  
model = load_model(model_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Basic Image Upscaler")
        self.root.geometry("500x700")  
        self.root.configure(bg="#e83e00")  
        self.root.resizable(width=False, height=False) 

        self.img_label = tk.Label(root, text="No Image Loaded")
        self.img_label.pack(pady=10)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image, fg="#080808", bg="#f7ae02")
        self.load_button.pack(pady=5)

        self.scale_label = tk.Label(root, text="Select Upscale Factor:")
        self.scale_label.pack(pady=5)

        self.scale_var = tk.IntVar(value=2)
        self.scale_menu = ttk.Combobox(root, textvariable=self.scale_var, values=[2, 4, 8])
        self.scale_menu.pack(pady=5)

        self.upscale_button = tk.Button(root, text="Upscale Image", command=self.confirm_upscale, fg="#080808", bg="#f7ae02")
        self.upscale_button.pack(pady=20)

        self.preview_label = tk.Label(root, text="Preview:")
        self.preview_label.pack(pady=5)

        self.image_preview = tk.Label(root)
        self.image_preview.pack(pady=10)

        self.loaded_image = None
        
        
        self.root.protocol("WM_DELETE_WINDOW", self.confirm_close)
        
        
        

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.loaded_image = Image.open(file_path).convert('RGB')
            resized_img = self.resize_image(self.loaded_image, (200, 200))
            img_tk = ImageTk.PhotoImage(resized_img)
            self.img_label.config(image=img_tk, text="")
            self.img_label.image = img_tk
            self.preview_image(self.loaded_image)

    def preview_image(self, img):
        resized_img = self.resize_image(img, (250, 250))
        img_tk = ImageTk.PhotoImage(resized_img)
        self.image_preview.config(image=img_tk, text="")
        self.image_preview.image = img_tk

    def resize_image(self, img, size):
    
        width, height = img.size
        max_width, max_height = size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return img.resize((new_width, new_height), Image.LANCZOS)

    def confirm_upscale(self):
        if not self.loaded_image:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        scale_factor = self.scale_var.get()
        if scale_factor not in [2, 4, 8]:
            messagebox.showwarning("Warning", "Invalid scale factor. Please choose between 2, 4, or 8.")
            return

        estimated_time = estimate_processing_time(model, self.loaded_image, scale_factor)
        if messagebox.askokcancel("Confirmation", f"The estimated processing time is {estimated_time:.2f} minutes. Do you want to continue?"):
            self.start_upscale_thread()

    def start_upscale_thread(self):
        thread = threading.Thread(target=self.show_progress)
        thread.start()

    def show_progress(self):
        scale_factor = self.scale_var.get()
        
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Upscaling Progress")
        progress_window.geometry("300x50")

        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=200, mode="determinate")
        progress_bar.pack(pady=10)

        
        upscaled_img = upscale_image(model, self.loaded_image, scale_factor, progress_bar)

        progress_window.destroy()  

        
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            upscaled_img.save(save_path)
            messagebox.showinfo("Information", f"Upscaled image saved at: {save_path}")
            
    def confirm_close(self):
        if messagebox.askokcancel("Confirm Close", "Are you sure you want to close the program?"):
            self.root.destroy()      
            
            
   
            
root = tk.Tk()
app = App(root)
root.mainloop()
