import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from data_processing import DataProcessor
from clustering import Clustering
from visualization import Visualization

class ClusteringApp(ttk.Window):
    def __init__(self):
        super().__init__(themename='superhero')
        self.title("Система кластеризации клиентов")
        self.geometry("1200x800")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # Обеспечивает корректное закрытие приложения

        # Инициализация компонентов
        self.data_processor = DataProcessor()
        self.clustering = Clustering()
        self.visualization = Visualization()

        # Данные
        self.loaded_data = None
        self.scaled_data = None
        self.data_columns = None
        self.clusters = None  # Массив с метками кластеров

        # Создание меню
        self.create_menu()

        # Создание вкладок
        self.create_tabs()

        # Привязка событий
        self.cluster_method.bind("<<ComboboxSelected>>", self.on_method_change)

    def on_closing(self):
        self.destroy()
        self.quit()

    def create_menu(self):
        menubar = ttk.Menu(self)
        self.config(menu=menubar)

        file_menu = ttk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить данные", command=self.load_data)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.on_closing)

        help_menu = ttk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)

    def create_tabs(self):
        # Создание вкладок
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Вкладка 1: Загрузка данных
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text='Загрузка данных')
        self.create_tab1_widgets()

        # Вкладка 2: Определение количества кластеров
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text='Определение количества кластеров')
        self.create_tab2_widgets()

        # Вкладка 3: Кластеризация и визуализация
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text='Кластеризация и визуализация')
        self.create_tab3_widgets()

        # Вкладка 4: Результаты и статистика
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text='Результаты и статистика')
        self.create_tab4_widgets()

    def create_tab1_widgets(self):
        # Элементы управления на вкладке 1
        frame = ttk.Frame(self.tab1)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.load_button = ttk.Button(frame, text="Загрузить данные", command=self.load_data, bootstyle="success")
        self.load_button.pack(pady=20)

        # Создаем фрейм для таблицы и скроллбаров
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill='both', expand=True)

        # Таблица для отображения данных
        self.data_treeview = ttk.Treeview(tree_frame)
        self.data_treeview.pack(side='left', fill='both', expand=True)

        # Скроллбары
        scrollbar_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.data_treeview.yview)
        scrollbar_y.pack(side='right', fill='y')

        scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=self.data_treeview.xview)
        scrollbar_x.pack(side='bottom', fill='x')

        self.data_treeview.configure(yscroll=scrollbar_y.set, xscroll=scrollbar_x.set)

    def create_tab2_widgets(self):
        frame = ttk.Frame(self.tab2)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.elbow_button = ttk.Button(frame, text="Определить оптимальное число кластеров", command=self.elbow_method, bootstyle="primary")
        self.elbow_button.pack(pady=20)

        # Метка для отображения оптимального количества кластеров
        self.optimal_clusters_label = ttk.Label(frame, text="Оптимальное количество кластеров: ")
        self.optimal_clusters_label.pack(pady=5)

    def create_tab3_widgets(self):
        frame = ttk.Frame(self.tab3)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.cluster_method_label = ttk.Label(frame, text="Метод кластеризации:")
        self.cluster_method_label.grid(row=0, column=0, pady=5, sticky='e')

        self.cluster_method = ttk.Combobox(frame, values=["K-Means", "DBSCAN", "SOM"], state='readonly')
        self.cluster_method.grid(row=0, column=1, pady=5, sticky='w')
        self.cluster_method.current(0)

        self.cluster_label = ttk.Label(frame, text="Количество кластеров:")
        self.cluster_label.grid(row=1, column=0, pady=5, sticky='e')

        self.cluster_entry = ttk.Entry(frame)
        self.cluster_entry.grid(row=1, column=1, pady=5, sticky='w')

        # Параметры DBSCAN
        self.eps_label = ttk.Label(frame, text="eps (для DBSCAN):")
        self.eps_entry = ttk.Entry(frame)
        self.min_samples_label = ttk.Label(frame, text="min_samples (для DBSCAN):")
        self.min_samples_entry = ttk.Entry(frame)

        # Параметры SOM
        self.som_size_label = ttk.Label(frame, text="Размер карты SOM:")
        self.som_size_entry = ttk.Entry(frame)
        self.som_iterations_label = ttk.Label(frame, text="Итерации SOM:")
        self.som_iterations_entry = ttk.Entry(frame)

        # По умолчанию скрываем дополнительные параметры
        self.eps_label.grid_forget()
        self.eps_entry.grid_forget()
        self.min_samples_label.grid_forget()
        self.min_samples_entry.grid_forget()
        self.som_size_label.grid_forget()
        self.som_size_entry.grid_forget()
        self.som_iterations_label.grid_forget()
        self.som_iterations_entry.grid_forget()

        self.cluster_button = ttk.Button(frame, text="Выполнить кластеризацию", command=self.perform_clustering, bootstyle="success")
        self.cluster_button.grid(row=6, column=0, columnspan=2, pady=20)

        # Прогресс-бар
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=200, mode="indeterminate")
        self.progress_bar.grid(row=7, column=0, columnspan=2, pady=10)

    def create_tab4_widgets(self):
        frame = ttk.Frame(self.tab4)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Кнопка для сохранения результатов
        self.save_button = ttk.Button(frame, text="Сохранить результаты", command=self.save_results, bootstyle="secondary")
        self.save_button.pack(pady=10)

        # Создаем фрейм для таблицы и скроллбаров
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill='both', expand=True)

        # Таблица для отображения результатов
        self.result_treeview = ttk.Treeview(tree_frame)
        self.result_treeview.pack(side='left', fill='both', expand=True)

        # Скроллбары
        scrollbar_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.result_treeview.yview)
        scrollbar_y.pack(side='right', fill='y')

        scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=self.result_treeview.xview)
        scrollbar_x.pack(side='bottom', fill='x')

        self.result_treeview.configure(yscroll=scrollbar_y.set, xscroll=scrollbar_x.set)

        # Метка для отображения коэффициента силуэта
        self.silhouette_label = ttk.Label(frame, text="Средний коэффициент силуэта: ")
        self.silhouette_label.pack(pady=5)

        # Кнопка для отображения графика коэффициентов силуэта
        self.silhouette_button = ttk.Button(frame, text="Показать график коэффициентов силуэта", command=self.show_silhouette_plot, bootstyle="info")
        self.silhouette_button.pack(pady=10)

        # Кнопка для отображения статистики по кластерам
        self.cluster_stats_button = ttk.Button(frame, text="Показать статистику по кластерам", command=self.show_cluster_stats, bootstyle="info")
        self.cluster_stats_button.pack(pady=10)

    def show_about(self):
        messagebox.showinfo("О программе", "Система кластеризации клиентов\nВерсия 1.2")

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if not file_path:
            messagebox.showwarning("Предупреждение", "Файл не выбран")
            return
        data = self.data_processor.load_data(file_path)
        if data is not None:
            self.loaded_data = data
            columns_list = list(self.loaded_data.columns)
            messagebox.showinfo("Заголовки столбцов", f"Загруженные столбцы: {columns_list}")
            self.display_data_in_treeview(self.loaded_data)

    def display_data_in_treeview(self, data):
        treeview = self.data_treeview
        treeview.delete(*treeview.get_children())

        treeview["columns"] = list(data.columns)
        treeview["show"] = "headings"

        for col in data.columns:
            treeview.heading(col, text=col)
            max_width = max(data[col].astype(str).map(len).max(), len(col))
            treeview.column(col, width=min(max_width * 10, 200), minwidth=100, stretch=True)

        for _, row in data.iterrows():
            treeview.insert("", "end", values=list(row))

    def on_method_change(self, event):
        method = self.cluster_method.get()
        # Скрываем все дополнительные параметры
        self.cluster_entry.grid_forget()
        self.cluster_label.grid_forget()
        self.eps_label.grid_forget()
        self.eps_entry.grid_forget()
        self.min_samples_label.grid_forget()
        self.min_samples_entry.grid_forget()
        self.som_size_label.grid_forget()
        self.som_size_entry.grid_forget()
        self.som_iterations_label.grid_forget()
        self.som_iterations_entry.grid_forget()

        if method == 'K-Means':
            self.cluster_label.config(text="Количество кластеров:")
            self.cluster_label.grid(row=1, column=0, pady=5, sticky='e')
            self.cluster_entry.grid(row=1, column=1, pady=5, sticky='w')
            self.cluster_entry.config(state='normal')
        elif method == 'DBSCAN':
            self.eps_label.grid(row=1, column=0, pady=5, sticky='e')
            self.eps_entry.grid(row=1, column=1, pady=5, sticky='w')
            self.min_samples_label.grid(row=2, column=0, pady=5, sticky='e')
            self.min_samples_entry.grid(row=2, column=1, pady=5, sticky='w')
        elif method == 'SOM':
            self.som_size_label.grid(row=1, column=0, pady=5, sticky='e')
            self.som_size_entry.grid(row=1, column=1, pady=5, sticky='w')
            self.som_iterations_label.grid(row=2, column=0, pady=5, sticky='e')
            self.som_iterations_entry.grid(row=2, column=1, pady=5, sticky='w')

    def elbow_method(self):
        if self.loaded_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return
        result = self.data_processor.preprocess_data(self.loaded_data)
        if result is None:
            return
        self.scaled_data, self.data_columns = result
        self.progress_bar.start()
        self.after(100, self.calculate_elbow_method)

    def calculate_elbow_method(self):
        scaled_data = self.scaled_data
        K, inertia, silhouette_scores = self.clustering.calculate_elbow_method(scaled_data)
        self.visualization.plot_elbow_method(K, inertia, self.tab2, 'canvas_elbow')
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        self.optimal_clusters_label.config(text=f"Оптимальное количество кластеров: {optimal_clusters}")
        self.progress_bar.stop()

    def perform_clustering(self):
        if self.loaded_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return
        result = self.data_processor.preprocess_data(self.loaded_data)
        if result is None:
            return
        self.scaled_data, self.data_columns = result

        clustering_method = self.cluster_method.get()
        scaled_data = self.scaled_data
        columns = self.data_columns

        self.progress_bar.start()
        self.after(100, lambda: self.run_clustering(clustering_method, scaled_data, columns))

    def run_clustering(self, method, scaled_data, columns):
        if method == 'K-Means':
            try:
                n_clusters = int(self.cluster_entry.get())
                if n_clusters <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ошибка", "Введите корректное количество кластеров")
                self.progress_bar.stop()
                return
            clusters = self.clustering.kmeans_clustering(scaled_data, n_clusters)
            self.clusters = clusters
            self.visualization.visualize_clusters(scaled_data, clusters, columns, self.tab3, 'canvas')
        elif method == 'DBSCAN':
            try:
                eps = float(self.eps_entry.get()) if self.eps_entry.get() else 0.5
                min_samples = int(self.min_samples_entry.get()) if self.min_samples_entry.get() else 5
                if eps <= 0 or min_samples <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ошибка", "Введите корректные параметры для DBSCAN")
                self.progress_bar.stop()
                return
            clusters = self.clustering.dbscan_clustering(scaled_data, eps, min_samples)
            self.clusters = clusters
            self.visualization.visualize_clusters(scaled_data, clusters, columns, self.tab3, 'canvas')
        elif method == 'SOM':
            try:
                som_size = int(self.som_size_entry.get()) if self.som_size_entry.get() else 10
                som_iterations = int(self.som_iterations_entry.get()) if self.som_iterations_entry.get() else 100
                if som_size <= 0 or som_iterations <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ошибка", "Введите корректные параметры для SOM")
                self.progress_bar.stop()
                return
            positions = self.clustering.som_clustering(scaled_data, som_size, som_iterations)
            # Преобразуем позиции в метки кластеров
            unique_positions, labels = np.unique(positions, axis=0, return_inverse=True)
            self.clusters = labels
            self.visualization.visualize_som(positions, labels, som_size, self.tab3, 'canvas')
        else:
            messagebox.showerror("Ошибка", "Неизвестный метод кластеризации")
            self.progress_bar.stop()
            return

        # Отображаем результаты
        self.display_results()

        self.progress_bar.stop()

    def display_results(self):
        # Обновляем таблицу с результатами
        if self.loaded_data is not None and self.clusters is not None:
            data_with_clusters = self.loaded_data.copy()
            data_with_clusters['Cluster'] = self.clusters
            self.display_result_treeview(data_with_clusters)

            # Рассчитываем коэффициент силуэта
            if len(set(self.clusters)) > 1 and -1 not in set(self.clusters):
                silhouette_avg = self.clustering.calculate_silhouette(self.scaled_data, self.clusters)
                self.silhouette_label.config(text=f"Средний коэффициент силуэта: {silhouette_avg:.4f}")
            else:
                self.silhouette_label.config(text="Коэффициент силуэта не может быть рассчитан для данных кластеров")

    def display_result_treeview(self, data):
        treeview = self.result_treeview
        treeview.delete(*treeview.get_children())

        treeview["columns"] = list(data.columns)
        treeview["show"] = "headings"

        for col in data.columns:
            treeview.heading(col, text=col)
            max_width = max(data[col].astype(str).map(len).max(), len(col))
            treeview.column(col, width=min(max_width * 10, 200), minwidth=100, stretch=True)

        for _, row in data.iterrows():
            treeview.insert("", "end", values=list(row))

    def save_results(self):
        if self.loaded_data is not None and self.clusters is not None:
            data_with_clusters = self.loaded_data.copy()
            data_with_clusters['Cluster'] = self.clusters
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                data_with_clusters.to_csv(file_path, index=False)
                messagebox.showinfo("Сохранение", f"Результаты сохранены в {file_path}")
        else:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")

    def show_silhouette_plot(self):
        if self.scaled_data is not None and self.clusters is not None:
            if len(set(self.clusters)) > 1 and -1 not in set(self.clusters):
                self.visualization.plot_silhouette(self.scaled_data, self.clusters, self.tab4, 'canvas_silhouette')
            else:
                messagebox.showwarning("Предупреждение", "Коэффициент силуэта не может быть рассчитан для данных кластеров")
        else:
            messagebox.showwarning("Предупреждение", "Сначала выполните кластеризацию")

    def show_cluster_stats(self):
        if self.loaded_data is not None and self.clusters is not None:
            data_with_clusters = self.loaded_data.copy()
            data_with_clusters['Cluster'] = self.clusters
            stats = data_with_clusters.groupby('Cluster').mean()
            self.visualization.show_cluster_statistics(stats)
        else:
            messagebox.showwarning("Предупреждение", "Сначала выполните кластеризацию")
