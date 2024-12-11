import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples
import ttkbootstrap as ttk  # Добавлен импорт ttk


class Visualization:
    def display_plot(self, fig, frame, canvas_attr):
        if hasattr(frame, canvas_attr):
            getattr(frame, canvas_attr).get_tk_widget().pack_forget()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        setattr(frame, canvas_attr, canvas)

    def visualize_clusters(self, scaled_data, clusters, columns, frame, canvas_attr):
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        df['Cluster'] = clusters
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2', ax=ax)
        ax.set_title('Кластеризация клиентов')
        self.display_plot(fig, frame, canvas_attr)

    def plot_elbow_method(self, K, inertia, frame, canvas_attr):
        fig, ax1 = plt.subplots(figsize=(10, 8))
        ax1.plot(K, inertia, 'bx-', label='Inertia (Метод локтя)')
        ax1.set_xlabel('Количество кластеров')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Метод локтя')
        self.display_plot(fig, frame, canvas_attr)

    def visualize_som(self, positions, clusters, som_size, frame, canvas_attr):
        fig, ax = plt.subplots(figsize=(10, 8))
        x = [pos[0] + 0.5 for pos in positions]
        y = [pos[1] + 0.5 for pos in positions]
        scatter = ax.scatter(x, y, c=clusters, cmap='tab20', s=50, alpha=0.7)
        ax.set_xlim([0, som_size])
        ax.set_ylim([0, som_size])
        ax.set_title('Распределение данных на карте SOM')
        ax.invert_yaxis()
        self.display_plot(fig, frame, canvas_attr)

    def plot_silhouette(self, scaled_data, clusters, frame, canvas_attr):
        cluster_labels = np.unique(clusters)
        n_clusters = cluster_labels.shape[0]

        silhouette_vals = silhouette_samples(scaled_data, clusters)
        y_lower, y_upper = 0, 0
        yticks = []

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[clusters == c]
            c_silhouette_vals.sort()
            y_upper += len(c_silhouette_vals)
            ax.barh(range(y_lower, y_upper), c_silhouette_vals, edgecolor='none', height=1)
            yticks.append((y_lower + y_upper) / 2)
            y_lower += len(c_silhouette_vals)

        ax.axvline(silhouette_vals.mean(), color="red", linestyle="--")
        ax.set_yticks(yticks)
        ax.set_yticklabels(cluster_labels)
        ax.set_ylabel('Кластер')
        ax.set_xlabel('Коэффициент силуэта')
        ax.set_title('График коэффициентов силуэта для кластеров')
        self.display_plot(fig, frame, canvas_attr)

    def show_cluster_statistics(self, stats):
        # Окно с таблицей статистики
        from tkinter import Toplevel
        top = Toplevel()
        top.title("Статистика по кластерам")
        top.geometry("800x600")

        treeview = ttk.Treeview(top)
        treeview.pack(fill='both', expand=True)

        treeview["columns"] = list(stats.columns)
        treeview["show"] = "headings"

        treeview.heading("#0", text="Cluster")
        treeview.column("#0", width=100)

        for col in stats.columns:
            treeview.heading(col, text=col)
            treeview.column(col, width=100, minwidth=100, stretch=True)

        for idx, row in stats.iterrows():
            treeview.insert("", "end", text=str(idx), values=list(row))

        scrollbar_y = ttk.Scrollbar(top, orient="vertical", command=treeview.yview)
        scrollbar_y.pack(side='right', fill='y')

        scrollbar_x = ttk.Scrollbar(top, orient="horizontal", command=treeview.xview)
        scrollbar_x.pack(side='bottom', fill='x')

        treeview.configure(yscroll=scrollbar_y.set, xscroll=scrollbar_x.set)
