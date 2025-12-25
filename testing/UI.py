import tkinter as tk
import os
import pandas as pd

#load data
dir_path = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(dir_path, "data/medical_questions_answered.parquet")
data = pd.read_parquet(FILE)

COLUMNS = ["question", "answer", "LLM_answer", "LLM_context"]
HEIGHTS = {
    "question": 4,
    "answer": 4,
    "LLM_answer": 4,
    "LLM_context": 16,
}
QUESTION_TYPES = sorted(data["question_type"].dropna().unique())


def load_random_row():
    qtype = selected_qtype.get()
    if qtype:
        subset = data[data["question_type"] == qtype]
    else:
        subset = data
    if subset.empty:
        return
    row = subset.sample(1).iloc[0]

    for col, text_widget in text_boxes.items():
        text_widget.config(state="normal")
        text_widget.delete("1.0", tk.END)
        value = "" if pd.isna(row[col]) else str(row[col])
        if col == "LLM_context":
            sep = "\n\n" + "-" * 100 + "\n\n"
            parts = value.split(sep)
            parts = [f"Context {i+1}/{len(parts)}:\n{parts[i]}" for i in range(len(parts))]
            value = sep.join(parts)
        text_widget.insert(tk.END, value)
        text_widget.config(state="disabled")

root = tk.Tk()
root.title("Random QA Viewer")
root.geometry("900x700")
root.resizable(True, True)
selected_qtype = tk.StringVar()
selected_qtype.set(QUESTION_TYPES[0])

text_boxes = {}
for col in COLUMNS:
    label = tk.Label(root, text=col.upper(), font=("Arial", 10, "bold"))
    label.pack(anchor="w", padx=10, pady=(10, 0))

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    text = tk.Text(
        frame,
        height=HEIGHTS[col],
        wrap="word",
        yscrollcommand=scrollbar.set
    )
    text.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=text.yview)

    text.config(state="disabled")
    text_boxes[col] = text

# dropdown
row = tk.Frame(root)
row.pack(anchor="w", padx=10, pady=(10, 0))
tk.Label(row, text="QUESTION TYPE:", font=("Arial", 10, "bold")).pack(side="left")
dropdown = tk.OptionMenu(row, selected_qtype, *QUESTION_TYPES)
dropdown.pack(side="left", padx=10)

btn = tk.Button(root, text="Load Random Row", command=load_random_row)
btn.pack(pady=10)

load_random_row()
root.mainloop()
