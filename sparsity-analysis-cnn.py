import numpy as np
import random

random.seed(123)
RUNS=100

# Input feature map shape.
CI, HI, WI = 512, 14, 14
STRIDE = 1
PAD = 0
# Kernel weights shape.
KSZ = 3

# Output feature map shape.
HO = int((HI + 2 * PAD - KSZ) / STRIDE) + 1
WO = int((WI + 2 * PAD - KSZ) / STRIDE) + 1
CO = 1

# Energies taken from Horowitz'14
ENERGIES = dict(add=0.03, mult=0.2, memory=2.5)

weights = np.empty((CO, CI, KSZ, KSZ))
state_init = np.empty((CO, HO, WO))
for co in range(CO):
    for ho in range(HO):
        for wo in range(WO):
            state_init[co][ho][wo] = random.randint(-128, 63)
    for ci in range(CI):
        for hk in range(KSZ):
            for wk in range(KSZ):
                weights[co][ci][hk][wk] = random.randint(-128, 127)

ifmap_snn = np.empty((CI, HI, WI), dtype=int)
ifmap_ann = np.empty((CI, HI, WI), dtype=int)

# sparsities = np.array(
#     [0, 0.2, 0.4, 0.6, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.95, 0.96, 0.97, 0.98, 
#      0.99]
#     )

sparsities = np.array(
    [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    )

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
        for ci in range(CI):
            for hi in range(HI):
                for wi in range(WI):
                    val = int(random.uniform(0, 1) > sparsity)
                    ifmap_snn[ci][hi][wi] = val
                    ifmap_ann[ci][hi][wi] = random.randint(-128, 127) * val

        # SNN state.
        state = state_init.copy()
        for co in range(CO):
            for ho in range(HO):
                for wo in range(WO):
                    buff_snn, buff_ann = 0, 0
                    got_spike = False
                    for ci in range(CI):
                        for hk in range(KSZ):
                            for wk in range(KSZ):
                                wi = wo * STRIDE + wk
                                hi = ho * STRIDE + hk
                                if ifmap_snn[ci][hi][wi] == 1:
                                    snn_energy["mem"][i] += ENERGIES["memory"] / 8
                                    got_spike = True
                                    snn_energy["mem"][i] += ENERGIES["memory"]
                                    buff_snn += weights[co][ci][hk][wk]
                                    snn_energy["comp"][i] += ENERGIES["add"]

                                if ifmap_ann[ci][hi][wi] != 0:
                                    ann_energy["mem"][i] += ENERGIES["memory"]
                                    ann_energy["mem"][i] += ENERGIES["memory"]
                                    ann_energy["comp"][i] += ENERGIES["mult"]
                                    buff_ann += (
                                        ifmap_ann[ci][hi][wi] * weights[co][ci][hk][wk]
                                    )
                                    ann_energy["comp"][i] += ENERGIES["add"]

                    if got_spike:
                        # Read state
                        mem = state[co][ho][wo]
                        snn_energy["mem"][i] += ENERGIES["memory"]
                        # Apply leakage.
                        mem *= 0.95
                        snn_energy["comp"][i] += ENERGIES["mult"]
                        # Accumulate activation.
                        mem += buff_snn
                        snn_energy["comp"][i] += ENERGIES["add"]
                        # Thresholding.
                        if mem > 64:
                            # Subtract threshold.
                            mem -= 64
                            snn_energy["comp"][i] += ENERGIES["add"]
                            out_spike = 1
                        # Write back the membrane.
                        snn_energy["mem"][i] += ENERGIES["memory"]
                    # Write output spike.
                    snn_energy["mem"][i] += ENERGIES["memory"] / 8
                    ann_energy["mem"][i] += ENERGIES["memory"]  

    # Avearage energies along the runs.
    ann_energy["mem"][i] /= RUNS
    snn_energy["mem"][i] /= RUNS
    ann_energy["comp"][i] /= RUNS
    snn_energy["comp"][i] /= RUNS
    ann_energy["tot"][i] = ann_energy["mem"][i] + ann_energy["comp"][i]
    snn_energy["tot"][i] = snn_energy["mem"][i] + snn_energy["comp"][i]

    print("SNN:")
    print(f"\t- Total energy: {snn_energy['tot'][i]/1e3:.2f} uJ.")
    print(f"\t- Memory energy: {snn_energy['mem'][i]/1e3:.2f} uJ.")
    print(f"\t- Computations energy: {snn_energy['comp'][i]/1e3:.2f} uJ.")
    print("ANN:")
    print(f"\t- Total energy: {ann_energy['tot'][i]/1e3:.2f} uJ.")
    print(f"\t- Memory energy: {ann_energy['mem'][i]/1e3:.2f} uJ.")
    print(f"\t- Computations energy: {ann_energy['comp'][i]/1e3:.2f} uJ.")

PRINT_FOR_LATEX = False
if PRINT_FOR_LATEX:
    print("*"*50)
    for i, s in enumerate(sparsities):
        print(f"({s*100:.0f}, {ann_energy['tot'][i]/1e3:.2f})")
    print("*"*50)
    for i, s in enumerate(sparsities):
        print(f"({s*100:.0f}, {snn_energy['tot'][i]/1e3:.2f})")
    print("*"*50)
    for i, s in enumerate(sparsities):
        print(f"({s*100:.0f}, {ann_energy['mem'][i]/1e3:.2f})")
    print("*"*50)
    for i, s in enumerate(sparsities):
        print(f"({s*100:.0f}, {snn_energy['mem'][i]/1e3:.2f})")
    print("*"*50)
    for i, s in enumerate(sparsities):
        print(f"({s*100:.0f}, {ann_energy['comp'][i]/1e3:.2f})")
    print("*"*50)
    for i, s in enumerate(sparsities):
        print(f"({s*100:.0f}, {snn_energy['comp'][i]/1e3:.2f})")

PLOT = False
if PLOT:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

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
    fig.add_trace(traces[0], row=1, col=1)
    fig.add_trace(traces[1], row=1, col=1)
    fig.add_trace(traces[2], row=1, col=2)
    fig.add_trace(traces[3], row=1, col=2)
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

    pio.write_image(fig, "energy-analysis-cnn.png")
    pio.write_image(fig, "energy-analysis-cnn.svg")
    fig.show()
