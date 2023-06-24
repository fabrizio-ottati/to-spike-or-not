import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(123)
RUNS=100


M, N = 512, 1024

ENERGIES = dict(add=0.03, mult=0.2, read=2.5)

weights = np.empty((M, N))
rec_weights = np.empty((M,))
snn_state_init = np.empty((M, ))
ann_state_init = np.empty((M, ))
for m in range(M):
    snn_state_init[m] = random.randint(-128, 63)
    ann_state_init[m] = random.randint(-128, 63)
    rec_weights[m] = random.randint(-128, 127)
    for n in range(N):
        weights[m][n] = random.randint(-128, 127)

ifmap_snn = np.empty((N,), dtype=int)
ifmap_ann = np.empty((N,), dtype=int)

# sparsities = np.array(
#     [0, 0.2, 0.4, 0.6, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.95, 0.96, 0.97, 0.98, 
#      0.99]
#         )
sparsities = np.array(
    [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    )
# sparsities = np.array([0.9, 0.95, 0.96, 0.97, 0.98, 0.99])
# sparsities = np.array([0])

snn_energy = dict(
    tot=np.zeros(sparsities.shape),
    mem=np.zeros(sparsities.shape),
    comp=np.zeros(sparsities.shape),
)
ann_energy = dict(
    tot=np.zeros(sparsities.shape),
    mem=np.zeros(sparsities.shape),
    comp=np.zeros(sparsities.shape),
)

for i, sparsity in enumerate(sparsities):
    print("*" * 50 + "\nSparsity: {}".format(int(sparsity * 100)) + "%.")
    for run in range(RUNS):

        # Generating random input values according to sparsity.
        for n in range(N):
            val = int(random.uniform(0, 1) > sparsity)
            ifmap_snn[n] = val
            ifmap_ann[n] = random.randint(-128, 127) * val

        snn_state = snn_state_init.copy()
        ann_state = ann_state_init.copy()
        for m in range(M):
            buff_snn, buff_ann = 0, 0
            got_spike = False
            got_non_zero = False
            for n in range(N):
                if ifmap_snn[n] == 1:
                    snn_energy["mem"][i] += ENERGIES["read"] / 8
                    got_spike = True
                    snn_energy["mem"][i] += ENERGIES["read"]
                    buff_snn += weights[m][n]
                    snn_energy["comp"][i] += ENERGIES["add"]

                if ifmap_ann[n] != 0:
                    ann_energy["mem"][i] += ENERGIES["read"]
                    got_non_zero = True
                    ann_energy["mem"][i] += ENERGIES["read"]
                    ann_energy["comp"][i] += ENERGIES["mult"]
                    ann_energy["comp"][i] += ENERGIES["add"]
                    buff_ann += (
                        ifmap_ann[n] * weights[m][n]
                    )

            if got_spike:
                mem = snn_state[m]
                snn_energy["mem"][i] += ENERGIES["read"]
                mem *= 0.95
                snn_energy["comp"][i] += ENERGIES["mult"]
                mem += buff_snn
                snn_energy["comp"][i] += ENERGIES["add"]
                if mem > 64:
                    mem -= 64
                    snn_energy["comp"][i] += ENERGIES["add"]
                    out_spike = 1
                    snn_energy["mem"][i] += ENERGIES["read"] / 8
                snn_energy["mem"][i] += ENERGIES["read"]

            if got_non_zero:
                ann_energy["mem"][i] += ENERGIES["read"]  # Write back the result.
                h = ann_state[m]
                ann_energy["mem"][i] += ENERGIES["read"]  # Write back the result.
                w = rec_weights[m]
                ann_energy["comp"][i] += ENERGIES["mult"]
                h += w*h + buff_ann
                ann_energy["comp"][i] += ENERGIES["add"]
                ann_state[m] = h
                ann_energy["mem"][i] += ENERGIES["read"]  # Write back the result.
                if buff_ann != 0:
                    ann_energy["mem"][i] += ENERGIES["read"]  # Write back the result.
        ann_energy["tot"][i] = ann_energy["mem"][i] + ann_energy["comp"][i]
        snn_energy["tot"][i] = snn_energy["mem"][i] + snn_energy["comp"][i]

    ann_energy["tot"][i] /= RUNS
    ann_energy["mem"][i] /= RUNS
    ann_energy["comp"][i] /= RUNS
    snn_energy["tot"][i] /= RUNS
    snn_energy["mem"][i] /= RUNS
    snn_energy["comp"][i] /= RUNS
    print("SNN:")
    print(f"\t- Total energy: {snn_energy['tot'][i]/1e3:.2f} uJ.")
    print(f"\t- Memory energy: {snn_energy['mem'][i]/1e3:.2f} uJ.")
    print(f"\t- Computations energy: {snn_energy['comp'][i]/1e3:.2f} uJ.")
    print("ANN:")
    print(f"\t- Total energy: {ann_energy['tot'][i]/1e3:.2f} uJ.")
    print(f"\t- Memory energy: {ann_energy['mem'][i]/1e3:.2f} uJ.")
    print(f"\t- Computations energy: {ann_energy['comp'][i]/1e3:.2f} uJ.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["Total energy", "Memory access energy", "Computation energy"],
    shared_xaxes=True,
    shared_yaxes=True,
)
# Create traces
traces = list()
line_traces = list()

ann_line = dict(
    tot=np.full(sparsities.shape, ann_energy["tot"][0]) * (1.0 - sparsities),
    mem=np.full(sparsities.shape, ann_energy["mem"][0]) * (1.0 - sparsities),
    comp=np.full(sparsities.shape, ann_energy["comp"][0]) * (1.0 - sparsities),
)

snn_line = dict(
    tot=np.full(sparsities.shape, snn_energy["tot"][0]) * (1.0 - sparsities),
    mem=np.full(sparsities.shape, snn_energy["mem"][0]) * (1.0 - sparsities),
    comp=np.full(sparsities.shape, snn_energy["comp"][0]) * (1.0 - sparsities),
)


# line_traces.append(
#     go.Scatter(
#         x=sparsities[4:] * 100,
#         y=ann_line["tot"][4:] / 1e6,
#         mode="lines",
#         showlegend=False,
#         line=dict(color="orange", width=4),
#     )
# )

# line_traces.append(
#     go.Scatter(
#         x=sparsities[4:] * 100,
#         y=snn_line["tot"][4:] / 1e6,
#         mode="lines",
#         showlegend=False,
#         line=dict(color="red", width=4),
#     )
# )

traces.append(
    go.Scatter(
        x=sparsities * 100,
        y=ann_energy["tot"] / 1e3,
        mode="markers",
        marker=dict(
            color="orange",
            symbol="cross-dot",
            line=dict(width=2, color="darkslategrey"),
            size=32,
        ),
        name="ANN",
        showlegend=True,
    )
)

traces.append(
    go.Scatter(
        x=sparsities * 100,
        y=snn_energy["tot"] / 1e3,
        mode="markers",
        marker=dict(
            color="red",
            symbol="x-dot",
            line=dict(width=2, color="darkslategrey"),
            size=32 ,
        ),
        name="SNN",
        showlegend=True,
    )
)

# line_traces.append(
#     go.Scatter(
#         x=sparsities[4:] * 100,
#         y=ann_line["mem"][4:] / 1e6,
#         mode="lines",
#         showlegend=False,
#         line=dict(color="orange", width=4),
#     )
# )

# line_traces.append(
#     go.Scatter(
#         x=sparsities[4:] * 100,
#         y=snn_line["mem"][4:] / 1e6,
#         mode="lines",
#         showlegend=False,
#         line=dict(color="red", width=4),
#     )
# )

traces.append(
    go.Scatter(
        x=sparsities * 100,
        y=ann_energy["mem"] / 1e3,
        mode="markers",
        marker=dict(
            color="orange",
            symbol="cross-dot",
            line=dict(width=2, color="darkslategrey"),
            size=32,
        ),
        showlegend=False,
    )
)

traces.append(
    go.Scatter(
        x=sparsities * 100,
        y=snn_energy["mem"] / 1e3,
        mode="markers",
        marker=dict(
            color="red",
            symbol="x-dot",
            line=dict(width=2, color="darkslategrey"),
            size=32,
        ),
        showlegend=False,
    )
)

# line_traces.append(
#     go.Scatter(
#         x=sparsities[4:] * 100,
#         y=ann_line["comp"][4:] / 1e6,
#         mode="lines",
#         showlegend=False,
#         line=dict(color="orange", width=4),
#     )
# )

# line_traces.append(
#     go.Scatter(
#         x=sparsities[4:] * 100,
#         y=snn_line["comp"][4:] / 1e6,
#         mode="lines",
#         showlegend=False,
#         line=dict(color="red", width=4),
#     )
# )

traces.append(
    go.Scatter(
        x=sparsities * 100,
        y=ann_energy["comp"] / 1e3,
        mode="markers",
        marker=dict(
            color="orange",
            symbol="cross-dot",
            line=dict(width=2, color="darkslategrey"),
            size=32,
        ),
        showlegend=False,
    )
)

traces.append(
    go.Scatter(
        x=sparsities * 100,
        y=snn_energy["comp"] / 1e3,
        mode="markers",
        marker=dict(
            color="red",
            symbol="x-dot",
            line=dict(width=2, color="darkslategrey"),
            size=32,
        ),
        showlegend=False,
    )
)

# Show the plot
# fig.add_trace(line_traces[0], row=1, col=1)
# fig.add_trace(line_traces[1], row=1, col=1)
fig.add_trace(traces[0], row=1, col=1)
fig.add_trace(traces[1], row=1, col=1)
# fig.add_trace(line_traces[2], row=1, col=2)
# fig.add_trace(line_traces[3], row=1, col=2)
fig.add_trace(traces[2], row=1, col=2)
fig.add_trace(traces[3], row=1, col=2)
# fig.add_trace(line_traces[4], row=1, col=3)
# fig.add_trace(line_traces[5], row=1, col=3)
fig.add_trace(traces[4], row=1, col=3)
fig.add_trace(traces[5], row=1, col=3)
fig.update_layout(
    title="Energy consumption v.s. input data sparsity",
    width=2500,
    height=1000,
    title_x=0.5,
    title_y=1.0,
    yaxis=dict(title="Energy [μJ]"),
    font=dict(family="Serif", size=38),
    legend=dict(itemsizing="trace", font=dict(size=38, family="Serif")),
)
fig.update_xaxes(title_text="Input data sparsity [%]", row=1, col=2)
fig.update_annotations(font_family="Serif", font_size=44)

import plotly.io as pio

# pio.write_image(fig, "energy-analysis.png")
# pio.write_image(fig, "energy-analysis.svg")
fig.show()
