import matplotlib.pyplot as plt
import numpy as np

F = open("Loss.txt", "r")
datas = F.read()
F.close()

Losses_keys = ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg', 'ROI_Loss']
Losses = [[],[],[],[],[],[]]

for data in datas.split("\n"):
    if data == "":
        break
    else:
        Loss = np.array(data.split(",")).astype(float)
        for idx, L in enumerate(Loss):
            Losses[idx].append(L)
        Losses[5].append(Loss[0] + Loss[1] + Loss[2])

Losses = np.array(Losses)
print(np.shape(Losses))
__, num = np.shape(Losses)


N = 570
for i in range(len(Losses_keys)):
    for j in range(num//N):
        Losses[i][j] = sum(Losses[i][j*N:(j+1)*N])/N

x = np.arange(0, num//N)
C = ['red', 'orange', 'green', 'blue', 'purple', 'fuchsia']
for idx, Losses_key in enumerate(Losses_keys):
    plt.plot(x, Losses[idx][:(num//N)], label=Losses_keys[idx], color = C[idx])
    # plt.plot(x[10:500], Losses[idx][10:500], label=Losses_keys[idx], color = C[idx])
plt.legend(Losses_keys)
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.grid(color = "k", linestyle = '--', linewidth = 0.3)
plt.show()
