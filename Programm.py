import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib

# Загрузка модели и векторизатора
model_package = joblib.load('model.pth')
models = model_package['models']
vectorizer = model_package['vectorizer']
categories = model_package['categories']

def predict_category():
    user_input = comment_text.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Предупреждение", "Пожалуйста, введите комментарий.")
        return
    comment_tok = ' '.join(user_input.lower().split())
    comment_vec = vectorizer.transform([comment_tok])
    predicted_probs = {}
    for category in categories:
        model = models.get(category)
        if model:
            proba = model.predict_proba(comment_vec)[0][1]
            predicted_probs[category] = proba
    if predicted_probs:
        predicted_category = max(predicted_probs, key=predicted_probs.get)
        confidence = predicted_probs[predicted_category]
        result_text = f"Комментарий:\n{user_input}\n\n" \
                      f"Категория: {predicted_category}\n" \
                      f"Уверенность: {confidence:.2f}"
        result_text_widget.config(state='normal')
        result_text_widget.delete("1.0", tk.END)
        result_text_widget.insert(tk.END, result_text)
        result_text_widget.config(state='disabled')
        # Автоматическая подгонка размеров окна
        root.update()
        root.minsize(root.winfo_width(), root.winfo_height())
    else:
        messagebox.showerror("Ошибка", "Не удалось сделать предсказание.")

# Создаем главное окно с большим размером
root = tk.Tk()
root.title("AI Инновационный классификатор")
root.geometry("1000x800")  # Увеличили размеры окна
root.configure(bg="#222831")

# Заголовок
title_label = tk.Label(root, text="🧠 AI Инновационный классификатор", font=("Helvetica Neue", 28, "bold"), bg="#222831", fg="#00FFCC")
title_label.pack(pady=20)

# Инструкция
instruction_label = tk.Label(root, text="Введите комментарий и нажмите 'Предсказать'", font=("Helvetica Neue", 18), bg="#222831", fg="#EEEEEE")
instruction_label.pack(pady=10)

# Поле для ввода
input_frame = tk.Frame(root, bg="#393E46", bd=2, relief=tk.RIDGE)
input_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

input_label = tk.Label(input_frame, text="Ваш комментарий:", font=("Helvetica Neue", 16), bg="#393E46", fg="#00FFCC")
input_label.pack(anchor='nw', padx=10, pady=5)

comment_text = tk.Text(input_frame, height=12, font=("Helvetica Neue", 16), wrap=tk.WORD, bg="#EEEEEE", fg="#222831", bd=0, highlightthickness=0)
comment_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Стиль для кнопки
style = ttk.Style()
style.theme_use('clam')
style.configure('TButton',
                font=('Helvetica Neue', 18, 'bold'),
                background='#00FFCC',
                foreground='#222831',
                padding=12)
style.map('TButton',
          background=[('active', '#00C8B4')],
          foreground=[('active', '#222831')])

# Кнопка предсказания
predict_button = ttk.Button(root, text="🔮 Предсказать", command=predict_category)
predict_button.pack(pady=20)

# Область для результата с прокруткой
result_container = tk.Frame(root, bg="#393E46")
result_container.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

# Скроллбар
scrollbar = tk.Scrollbar(result_container)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Текст для отображения результата
result_text_widget = tk.Text(result_container, height=10, font=("Helvetica Neue", 16), bg="#222831", fg="#00FFCC", wrap=tk.WORD, state='disabled', yscrollcommand=scrollbar.set)
result_text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

scrollbar.config(command=result_text_widget.yview)

# Запуск интерфейса
root.mainloop()