import tkinter as tk
from gui import ModernFaceRecognitionGUI
import logging
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    root = tk.Tk()
    app = ModernFaceRecognitionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        tk.messagebox.showerror("Critical Error", f"System error: {str(e)}")

if __name__ == "__main__":
    main()