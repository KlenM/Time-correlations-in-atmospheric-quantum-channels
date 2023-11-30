from matplotlib import pyplot as plt


plt.rcParams['axes.axisbelow'] = True
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = "STIX"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

LINE_MAIN_COLOR = "#363649"
LINE_COLORS = ["#16a9da", "#fe6361", "#ea8b06", "#9430b0", "#f7ca49", "#7d7059"]

SAVE_KWARGS = {
    "format": "pdf",
    "dpi": 300,
    "bbox_inches": "tight",
    "pad_inches": 0.005
    }
