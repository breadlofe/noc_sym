import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

methods = ['QAP', 'SA']
costs = [6574, 7059]
bar_labels = ['red', 'blue']
bar_colors = ['tab:red', 'tab:blue']

ax.bar(methods, costs, label=bar_labels, color=bar_colors)
ax.set_ylim(6000.0)
ax.set_ylabel('Total Hop Count Score')
ax.set_title('Total Hop Count Score Simulated Through Flow Matrix and Manhattan Distance')
#ax.legend(title='Method color')

# for p in ax.patches:
#     plt.errorbar(p.get_x()+p.get_width()/2., p.get_height(), yerr=0.0001, fmt='o', capsize=4)

for p in ax.patches:
    ax.annotate(text=np.round(p.get_height(), decimals=2),
                xy=(p.get_x()+p.get_width()/2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points')

plt.show()