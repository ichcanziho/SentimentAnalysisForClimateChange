import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.utils import UtilMethods


class Results(UtilMethods):

    def __init__(self):
        ...

    @UtilMethods.print_execution_time
    def make_freq(self, x_axes_file, y_axes_file, x_max_freq, y_max_freq,
                  sentiments, output_dir, scale=1, single_axes=True, interval=(0, -1)):

        def get_freq_window(only, word_col, freq_col):
            mask = [value in only for value in word_col]
            word_values = word_col
            word_values = word_values[mask]
            freq_values = freq_col
            freq_values = freq_values[mask]
            return word_values, freq_values

        x_ax = pd.read_csv(x_axes_file)
        y_ax = pd.read_csv(y_axes_file)
        counts = [col for col in x_ax.columns if 'counts' in col]
        words = [word for word in x_ax.columns if 'words' in word]

        if interval[1] == -1:
            x_ax = x_ax[interval[0]:]
            y_ax = y_ax[interval[0]:]
        else:
            x_ax = x_ax[interval[0]: interval[1]]
            y_ax = y_ax[interval[0]: interval[1]]

        x_ax.reset_index(drop=True, inplace=True)
        y_ax.reset_index(drop=True, inplace=True)
        for count in counts:
            sentiment = count.split("_")[0]
            x_freq = x_max_freq[sentiment]
            y_freq = y_max_freq[sentiment]
            x_ax[count] = x_ax[count] / x_freq * 100 * scale
            y_ax[count] = y_ax[count] / y_freq * 100 * scale

        matrix = []
        for word_col, count in zip(words, counts):
            sentiment = count.split("_")[0]

            if single_axes:
                only_on_x = [word for word in list(x_ax[word_col]) if word not in list(y_ax[word_col])]
                word_values, freq_values = get_freq_window(only_on_x, x_ax[word_col], x_ax[count])
                for w, f in zip(word_values, freq_values):
                    row = [w, f, 0, sentiments[sentiment]]
                    matrix.append(row)

                only_on_y = [word for word in list(y_ax[word_col]) if word not in list(x_ax[word_col])]
                word_values, freq_values = get_freq_window(only_on_y, y_ax[word_col], y_ax[count])
                for w, f in zip(word_values, freq_values):
                    row = [w, 0, f, sentiments[sentiment]]
                    matrix.append(row)

            both_on_x_y = [word for word in list(x_ax[word_col]) if word in list(y_ax[word_col])]
            word_values, freq_values_x = get_freq_window(both_on_x_y, x_ax[word_col], x_ax[count])
            word_values, freq_values_y = get_freq_window(both_on_x_y, y_ax[word_col], y_ax[count])
            for w, fx, fy in zip(word_values, freq_values_x, freq_values_y):
                row = [w, fx, fy, sentiments[sentiment]]
                matrix.append(row)

        frame = pd.DataFrame(data=matrix, columns=["WORDS", "X", "Y", "COLOR"])
        frame.to_csv(f'{output_dir}', index=False)

    @UtilMethods.print_execution_time
    def plot_test(self, input_file, save_dir, classifier, marker, size=(10, 10)):

        def putLegends(ax1, classifier, marker):
            feelings = {"positive": 'green', "neutral": "blue", "negative": "red"}
            for name, Color in feelings.items():
                ax1.plot(np.NaN, np.NaN, c=Color, label=name, marker='o', linestyle='None')
            ax2 = ax1.twinx()
            ax2.plot(np.NaN, np.NaN, label=classifier, c='black', marker=marker, linestyle='None')
            ax2.get_yaxis().set_visible(False)
            ax.legend(loc=4, title="Feelings")
            ax2.legend(loc=2, title='Model')

        frame = pd.read_csv(input_file)
        fig, ax = plt.subplots(figsize=size)

        for word, x, y, color in frame.values:
            ax.scatter(x, y, s=50, c=color, alpha=0.5, marker=marker)
            ax.text(x + 0.001, y + 0.001, word, fontsize=10)
        putLegends(ax, classifier, marker)
        plt.title(f'Most used words in tweets and news by sentiment using {classifier}')
        ax.set_xlabel("freq on twitter")
        ax.set_ylabel("freq on news")
        save_path = f'{save_dir}/{classifier}_summarize.svg'

        plt.savefig(save_path, bbox_inches='tight')
        plt.show()





