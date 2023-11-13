import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import simpson
from numpy import trapz

def PRplot(IOU_value, IOU, Names, category, filename):
    x = np.arange(0, 1.01, 0.01)
    # labels = []
    for idx, name in enumerate(Names):
        plt.plot(x, IOU[idx], label = f'model:{name}_IoU@{IOU_value} (area={round(trapz(IOU[idx], dx = 0.01), 3)})')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    # plt.legend(labels)
    plt.legend()
    plt.title(f'{filename} {category} IoU@{IOU_value} Precision Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color = "k", linestyle = '--', linewidth = 0.3)
    plt.savefig(f"./plot/{filename}_{category}_IoU@{IOU_value}_PR_Curvey.png")
    plt.show()

path = "./model/"
model_name = set([i.split(".")[0] for i in os.listdir(path) if i.endswith(".pth")])
path = "./plot/"
plot_name = set([i.split("_")[0] for i in os.listdir(path) if i.endswith(".npy")])
Names = list(plot_name & model_name)


for filename in ["BoundingBox", "Segmentation"]:
    for C_id, category in enumerate(["benign", "malignant"]):
        IOU_5 = []
        IOU_75 = []
        IOU_8 = []
        IOU_9 = []
        IOU_95 = []
        for name in Names:
            all_precision = np.load(os.path.join(path, f"{name}_{filename}_PR_Curvey.npy"))

            pr_5 = all_precision[0, :, C_id, 0, 2] # data for IoU@0.5
            pr_75 = all_precision[5, :, C_id, 0, 2] # data for IoU@0.75
            pr_8 = all_precision[6, :, C_id, 0, 2] # data for IoU@0.8
            pr_9 = all_precision[8, :, C_id, 0, 2] # data for IoU@0.9
            pr_95 = all_precision[9, :, C_id, 0, 2] # data for IoU@0.95

            IOU_5.append(pr_5)
            IOU_75.append(pr_75)
            IOU_8.append(pr_8)
            IOU_9.append(pr_9)
            IOU_95.append(pr_95)

        PRplot(IOU_value = 0.5, IOU = IOU_5, Names = Names, category = category, filename = filename)
        PRplot(IOU_value = 0.75, IOU = IOU_75, Names = Names, category = category, filename = filename)
        PRplot(IOU_value = 0.9, IOU = IOU_9, Names = Names, category = category, filename = filename)
# plt.plot(x, pr_9, label='IoU@0.9', color = 'blue')
# plt.plot(x, pr_95, label='IoU@0.9', color = 'blue')


# plt.savefig(f"{filename}_{category}_PR_Curvey.png")
# plt.show()

