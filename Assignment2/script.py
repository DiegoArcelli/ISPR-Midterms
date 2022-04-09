import matplotlib.pyplot as plt
from hmmlearn import hmm
import numpy as np
import pandas as pd

def get_model(data, n_states):
    # creation of the model specifying the algorithm to use for the decoding (viterbi)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", algorithm="viterbi", n_iter=1000)
    model.fit(data.reshape(-1,1))
    return model

def plot_models(data, models, column):
    colors = ["red", "green", "blue", "yellow", "purple"]
    for model in models:
        pred = model.predict(data.reshape(-1,1))
        for i in range(len(pred)):
            plt.plot(i, data[i], color=colors[pred[i]], marker="o")
        txt=f"{column}, model with {model.n_components} hidden states"
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        plt.show()

def compare_data(data, model, val_range, column):
    plt.plot(data, label="Actual data")
    mapping = np.linspace(val_range[0], val_range[1], model.n_components)
    pred = model.predict(data.reshape(-1,1))
    pred = [mapping[val] for val in pred]
    plt.plot(pred, label=f"{model.n_components} hidden states")
    txt=f"{column}, model with {model.n_components} hidden states"
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

dataset = pd.read_csv("energydata_complete.csv")
date = dataset["date"].to_numpy()
lights = dataset["lights"].to_numpy()
appliances = dataset["Appliances"].to_numpy()


n_states = [2, 3, 4, 5]

appliances_models = [get_model(appliances.reshape(-1,1), n_state) for n_state in n_states]
lights_models = [get_model(lights.reshape(-1,1), n_state) for n_state in n_states]


print("Appliances")
plot_models(appliances[:2900], appliances_models, "Appliances")
print("Lights")
plot_models(lights[:2900], lights_models, "Lights")


appliances_range = [appliances.min(), appliances.max()]
lights_range = [lights.min(), lights.max()]

for model in appliances_models:
    compare_data(appliances[:100], model, appliances_range, "Appliances")

for model in lights_models:
    compare_data(lights[:100], model, lights_range, "Lights")