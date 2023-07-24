import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

probscoretmp = pd.read_csv("predictedscore.csv", delimiter="\t", header=0)

probscore = probscoretmp[['Soft Voting model']]

print(probscore.keys())

plt.figure()

probscore.plot.hist(bins=50, alpha=0.8)
plt.xlabel('Predicted Probability Score')
plt.ylabel('Number of proteins')
plt.xlim(0.2,1.0)
plt.ylim(1,8000)
plt.legend(loc='upper left')
plt.grid(True) 
plt.show()
plt.savefig("human-protein-score-distribution.jpg", dpi=600)
