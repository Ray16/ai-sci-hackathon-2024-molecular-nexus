import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loss_csv = './logs/AI4Science/version_58/metrics.csv'
data = pd.read_csv(loss_csv)
data = data.dropna(subset=['val_loss'])
plt.plot(data['epoch'],data['val_loss'])
plt.xlabel('epoch')
plt.ylabel('val_loss')
plt.tight_layout()
plt.savefig('plot_loss.jpg',dpi=300,bbox_inches='tight')