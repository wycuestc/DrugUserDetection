import matplotlib.pyplot as plt
import numpy as np

x = [50, 100, 200, 300]

BLSTM_M1 = [0.722, 0.717, 0.716, 0.718]
BLSTM_M2 = [0.744, 0.723, 0.735, 0.729]
BLSTM_M3 = [0.748, 0.729, 0.734, 0.734]
Att_BLSTM_M1 = [0.731, 0.744, 0.742, 0.746]
Att_BLSTM_M2 = [0.761, 0.766, 0.757, 0.756]
Att_BLSTM_M3 = [0.750, 0.759, 0.755, 0.748]

plt.xticks(x)
axes = plt.gca()
axes.set_ylim([0,1])
#plt.yticks(np.arange(0,1, step=0.1))
plt.plot(x, BLSTM_M1, '-o', label = 'BLSTM-M1')
plt.plot(x, BLSTM_M2, '-o', label = 'BLSTM-M2')
plt.plot(x, BLSTM_M3, '-o', label = 'BLSTM-M3')
plt.plot(x, Att_BLSTM_M1, '-o', label = 'Att-BLSTM-M1')
plt.plot(x, Att_BLSTM_M2, '-o', label = 'Att-BLSTM-M2')
plt.plot(x, Att_BLSTM_M3, '-o', label = 'Att-BLSTM-M3')


plt.title('The influence of the dimension of word vectors')
plt.xlabel('The dimension of word vectors')
plt.ylabel('F1-score')
plt.legend(prop={'size': 8})
plt.show()