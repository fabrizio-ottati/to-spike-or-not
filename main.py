import pandas as pd

Ho, Wo, Co = 28, 28, 32
Hk, Wk, Ci = 3, 3, 144

ENERGY_MUL_8B = 0.2
ENERGY_ADD_8B = 0.03
ENERGY_MUL_32B = 3
ENERGY_ADD_32B = 0.1
ENERGY_MEM_8B = 100 / 8
ENERGY_MEM_32B = 100 / 2
ENERGY_MEM_1B = 100 / 64

CONV_OPS = Ho * Wo * Co * Hk * Wk * Ci
CONV_RD = Ho * Wo * Co * Hk * Wk * Ci
CONV_WR = Ho * Wo * Co

SEP = "-" * 80
# ------------------------------------------------------------------------------
ann = dict(ops=dict(), energy=dict())

# Multiplications involved in the convolution operation.
ann["ops"]["mul"] = CONV_OPS
# Each output pixel is dequantized.
ann["ops"]["quant"] = CONV_WR
# Same as mul but with bias addition at each output pixel.
ann["ops"]["add"] = CONV_OPS + CONV_WR
# Each output pixel passes through the activation.
ann["ops"]["nonLin"] = CONV_WR
# Number of weights needed for a convolution. Bias included (one per output channel).
ann["ops"]["weightRd"] = CONV_OPS + Co
# Number of inputs needed for a convolution.
ann["ops"]["ifmapRd"] = CONV_OPS
# Here we suppose that the quantization varies per output channel.
ann["ops"]["quantRd"] = Co
# Total number of memory read.
ann["ops"]["rd"] = sum([ann["ops"][k] for k in ("weightRd", "ifmapRd", "quantRd")])
# Total number of memory write.
ann["ops"]["wr"] = CONV_WR
# Total number of memory operations.
ann["ops"]["mem"] = ann["ops"]["wr"] + ann["ops"]["rd"]

ann["energy"]["mul"] = CONV_OPS * ENERGY_MUL_8B
ann["energy"]["quant"] = CONV_WR * ENERGY_MUL_32B
ann["energy"]["add"] = (CONV_OPS + CONV_WR) * ENERGY_ADD_32B
ann["energy"]["nonLin"] = CONV_WR * ENERGY_ADD_8B
ann["energy"]["weightRd"] = CONV_OPS * ENERGY_MEM_8B
ann["energy"]["ifmapRd"] = CONV_OPS * ENERGY_MEM_8B
ann["energy"]["quantRd"] = Co * ENERGY_MEM_32B
ann["energy"]["rd"] = CONV_OPS * 2 * ENERGY_MEM_8B + Co * 2 * ENERGY_MEM_32B
ann["energy"]["wr"] = CONV_WR * ENERGY_MEM_8B
ann["energy"]["mem"] = ann["energy"]["wr"] + ann["energy"]["rd"]
ann["energy"]["tot"] = sum(
    [ann["energy"][k] for k in ("mem", "nonLin", "add", "mul", "quant")]
)

ann["mem"] = dict()
ann["mem"]["params"] = Hk * Wk * Ci * 8 + Co * 2 * 32
ann["mem"]["tot"] = ann["mem"]["params"]


ann_pd = pd.DataFrame.from_dict(ann)
print(SEP)
print("ANN description:", end="\n\n")
print(ann_pd, end="\n\n")
print(f"ANN energy: {ann['energy']['tot'] * 1e-6:.2f} uJ.")
print(f"ANN memory occupation: {ann['mem']['tot'] / 1024:.2f} KiB.")

# ------------------------------------------------------------------------------
print(SEP)

snn = dict(ops=dict(), energy=dict())
snn["ops"]["mul"] = CONV_WR
snn["ops"]["add"] = CONV_OPS + CONV_WR + CONV_WR
snn["ops"]["nonLin"] = CONV_WR
snn["ops"]["weightRd"] = CONV_OPS + Co
snn["ops"]["ifmapRd"] = CONV_OPS
snn["ops"]["stateRd"] = CONV_WR
snn["ops"]["leakRd"] = Co
snn["ops"]["thresRd"] = Co
snn["ops"]["rd"] = sum(
    [snn["ops"][k] for k in ("weightRd", "ifmapRd", "stateRd", "leakRd", "thresRd")]
)
snn["ops"]["wr"] = CONV_WR * 2
# Total number of memory operations.
snn["ops"]["mem"] = snn["ops"]["wr"] + snn["ops"]["rd"]

snn["energy"]["mul"] = CONV_WR * ENERGY_MUL_32B
snn["energy"]["add"] = CONV_OPS * ENERGY_ADD_32B + CONV_WR * 2 * ENERGY_ADD_32B
snn["energy"]["nonLin"] = CONV_WR * ENERGY_ADD_32B
snn["energy"]["weightRd"] = CONV_OPS * ENERGY_MEM_8B + Co * ENERGY_MEM_32B
snn["energy"]["ifmapRd"] = CONV_OPS * ENERGY_MEM_1B
snn["energy"]["stateRd"] = CONV_WR * ENERGY_MEM_8B
snn["energy"]["leakRd"] = Co * ENERGY_MEM_32B
snn["energy"]["thresRd"] = Co * ENERGY_MEM_32B
snn["energy"]["rd"] = sum(
    [snn["energy"][k] for k in ("ifmapRd", "weightRd", "leakRd", "stateRd", "thresRd")]
)
snn["energy"]["wr"] = CONV_WR * (ENERGY_MEM_8B + ENERGY_MEM_1B)
snn["energy"]["mem"] = snn["energy"]["wr"] + snn["energy"]["rd"]
snn["energy"]["tot"] = sum([snn["energy"][k] for k in ("mul", "add", "nonLin", "mem")])

snn["mem"] = dict()
snn["mem"]["params"] = Hk * Wk * Ci * 8 + Co * 2 * 32
snn["mem"]["state"] = CONV_WR * 8
snn["mem"]["tot"] = snn["mem"]["params"] + snn["mem"]["state"]

snn_pd = pd.DataFrame.from_dict(snn)
print(SEP)
print("SNN description:", end="\n\n")
print(snn_pd, end="\n\n")
print(f"SNN energy: {snn['energy']['tot'] * 1e-6:.2f} uJ.")
print(f"SNN memory occupation: {snn['mem']['tot'] / 1024:.2f} KiB.")


# ------------------------------------------------------------------------------
print(SEP)
print(
    f"Energy ratio (SNN/ANN): {snn['energy']['tot'] / ann['energy']['tot'] * 100:.2f} %."
)
print(f"Memory ratio (SNN/ANN): {snn['mem']['tot'] / ann['mem']['tot'] * 100:.2f} %.")
print(SEP)
