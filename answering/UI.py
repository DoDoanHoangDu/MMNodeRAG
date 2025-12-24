import tkinter as tk
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
FILE1 = os.path.join(dir_path, "context.txt")
FILE2 = os.path.join(dir_path, "answer.txt")

def refresh():
    text1.config(state=tk.NORMAL)
    text2.config(state=tk.NORMAL)

    text1.delete("1.0", tk.END)
    text2.delete("1.0", tk.END)

    try:
        with open(FILE1, "r", encoding="utf-8") as f:
            text1.insert(tk.END, f.read())
    except IOError:
        pass

    try:
        with open(FILE2, "r", encoding="utf-8") as f:
            text2.insert(tk.END, f.read())
    except IOError:
        pass

    text1.config(state=tk.DISABLED)
    text2.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Answer")
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(1, weight=1)

tk.Label(root, text="Context").grid(row=0, column=0, sticky="nw", padx=5, pady=5)
tk.Label(root, text="Answer").grid(row=1, column=0, sticky="nw", padx=5, pady=5)

text1 = tk.Text(root, width=60, height=10, state=tk.DISABLED, wrap="word")
scroll1 = tk.Scrollbar(root, command=text1.yview)
text1.config(yscrollcommand=scroll1.set)

text1.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
scroll1.grid(row=0, column=2, sticky="ns")

text2 = tk.Text(root, width=60, height=10, state=tk.DISABLED, wrap="word")
scroll2 = tk.Scrollbar(root, command=text2.yview)
text2.config(yscrollcommand=scroll2.set)

text2.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
scroll2.grid(row=1, column=2, sticky="ns")

tk.Button(root, text="Refresh", command=refresh)\
    .grid(row=2, column=0, columnspan=3, pady=10)

refresh()
root.mainloop()



