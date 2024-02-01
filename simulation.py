from functools import partial
from pathlib import Path 

import numpy as np
import pandas as pd
import pyatmosphere as pyatm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


pyatm.gpu.config['use_gpu'] = True # set False if you don't have a GPU


CHANNELS = [
    {
        "name": "weak",
        "wind_speed": 10,
        "times": [0., 0.0008, 0.0016, 0.0024, 0.0032, 0.004 , 0.0048, 0.0056, 0.0064],
        "aperture_radiuses": [0.0006, 0.0008, 0.001, 0.0015, 0.002, 0.0025, 0.0035, 0.0045, 0.006, 0.008],
        "iterations": 25000,
        "save_step": 1000,
        "channel": pyatm.Channel(
            grid=pyatm.RectGrid(resolution=2**9, delta=0.0003),
            source=pyatm.GaussianSource(wvl=809e-9, w0=0.02, F0=1e3),
            path=pyatm.IdenticalPhaseScreensPath(
                phase_screen=pyatm.SSPhaseScreen(
                    model=pyatm.MVKModel(Cn2=5e-15, l0=1e-3, L0=80),
                    f_grid=pyatm.RandLogPolarGrid(points=2**10, f_min=1 / 80 / 15, f_max=1 / 1e-3 * 2)
                ), length=1e3, count=10
            ),
            pupil=pyatm.CirclePupil(radius=0.01),
        )
    },
    {
        "name": "moderate",
        "wind_speed": 10,
        "times": [0., 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004],
        "aperture_radiuses": [0.001, 0.0015, 0.0021, 0.0031, 0.0045, 0.0066, 0.0097, 0.014, 0.021, 0.03],
        "iterations": 25000,
        "save_step": 1000,
        "channel": pyatm.Channel(
            grid=pyatm.RectGrid(resolution=2**9, delta=0.0004), 
            source=pyatm.GaussianSource(wvl=809e-9, w0=0.02, F0=1.6e3),
            path=pyatm.IdenticalPhaseScreensPath(
                phase_screen=pyatm.SSPhaseScreen(
                    model=pyatm.MVKModel(Cn2=1.5e-14, l0=1e-3, L0=80),
                    f_grid=pyatm.RandLogPolarGrid(points=2**10, f_min=1 / 80 / 15, f_max=1 / 1e-3 * 2)
                ), length=1.6e3, count=10
            ),
            pupil=pyatm.CirclePupil(radius=0.04),
        )
    },
    {
        "name": "strong_d2",
        "wind_speed": 10,
        "times": [0., 0.0028, 0.0056, 0.0084, 0.0112, 0.014, 0.0168, 0.0196, 0.0224, 0.0252, 0.028],
        "aperture_radiuses": [0.004, 0.006, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        "iterations": 25000,
        "save_step": 1000,
        "channel": pyatm.Channel(
            grid=pyatm.RectGrid(resolution=2**11, delta=0.0015),
                source=pyatm.GaussianSource(wvl=808e-9, w0=0.08, F0=50e3),
            path=pyatm.IdenticalPhaseScreensPath(
                phase_screen=pyatm.SSPhaseScreen(
                    model=pyatm.MVKModel(Cn2=1e-16, l0=1e-3, L0=80),
                    f_grid=pyatm.RandLogPolarGrid(points=2**10, f_min=1 / 80 / 15, f_max=1 / 1e-3 * 2)
                ), length=50e3, count=15),
            pupil=pyatm.CirclePupil(radius=0.2)
        )
    },
    {
        "name": "strong",
        "wind_speed": 10,
        "times": [0., 0.0028, 0.0056, 0.0084, 0.0112, 0.014, 0.0168, 0.0196, 0.0224, 0.0252, 0.028],
        "aperture_radiuses": [0.004, 0.006, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        "iterations": 25000,
        "save_step": 1000,
        "channel": pyatm.Channel(
            grid=pyatm.RectGrid(resolution=2**11, delta=0.0015),
                source=pyatm.GaussianSource(wvl=808e-9, w0=0.08, F0=50e3),
            path=pyatm.IdenticalPhaseScreensPath(
                phase_screen=pyatm.SSPhaseScreen(
                    model=pyatm.MVKModel(Cn2=2e-16, l0=1e-3, L0=80),
                    f_grid=pyatm.RandLogPolarGrid(points=2**10, f_min=1 / 80 / 15, f_max=1 / 1e-3 * 2)
                ), length=50e3, count=15),
            pupil=pyatm.CirclePupil(radius=0.2)
        )
    },
    {
        "name": "strong_x1_5",
        "wind_speed": 10,
        "times": [0., 0.0028, 0.0056, 0.0084, 0.0112, 0.014, 0.0168, 0.0196, 0.0224, 0.0252, 0.028],
        "aperture_radiuses": [0.004, 0.006, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        "iterations": 25000,
        "save_step": 1000,
        "channel": pyatm.Channel(
            grid=pyatm.RectGrid(resolution=2**11, delta=0.0015),
                source=pyatm.GaussianSource(wvl=808e-9, w0=0.08, F0=50e3),
            path=pyatm.IdenticalPhaseScreensPath(
                phase_screen=pyatm.SSPhaseScreen(
                    model=pyatm.MVKModel(Cn2=3e-16, l0=1e-3, L0=80),
                    f_grid=pyatm.RandLogPolarGrid(points=2**10, f_min=1 / 80 / 15, f_max=1 / 1e-3 * 2)
                ), length=50e3, count=15),
            pupil=pyatm.CirclePupil(radius=0.2)
        )
    },
    {
        "name": "strong_adhoc_bell",
        "wind_speed": 10,
        "times": list(np.linspace(0, 1.5 * 2, 11) / 1000),
        "aperture_radiuses": [0.004, 0.006, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        "iterations": 50000,
        "save_step": 1000,
        "channel": pyatm.Channel(
            grid=pyatm.RectGrid(resolution=2**11, delta=0.0015),
                source=pyatm.GaussianSource(wvl=808e-9, w0=0.08, F0=50e3),
            path=pyatm.IdenticalPhaseScreensPath(
                phase_screen=pyatm.SSPhaseScreen(
                    model=pyatm.MVKModel(Cn2=2e-16, l0=1e-3, L0=80),
                    f_grid=pyatm.RandLogPolarGrid(points=2**10, f_min=1 / 80 / 15, f_max=1 / 1e-3 * 2)
                ), length=50e3, count=15),
            pupil=pyatm.CirclePupil(radius=0.1)
        )
    },
]


class TimeCoherenceResult(pyatm.simulations.WindResult):
    def __init__(self, channel, shifts, pupil=None, *args, **kwargs):
        pupil = pupil or channel.pupil
        measures = kwargs.pop("measures", [pyatm.simulations.Measure(
            channel, "atmosphere", partial(self.append_pupil, pupil),
            pyatm.measures.eta, time=shifts, name=f"{pupil.radius}")])
        super().__init__(*args, channel=channel, measures=measures, **kwargs)

    def append_pupil(self, pupil, channel, output):
        init_pupil = channel.pupil
        channel.pupil = pupil
        output = channel.pupil.output(output)
        channel.pupil = init_pupil
        return output

    def plot_output(self):
        print(f"Iteration: {len(self.measures[0].data)}")


def load_tc(path, time):
    return pd.DataFrame(
        [[float(v.strip()) for v in r[0][1:-2].split(',')] for r in pd.read_csv(path).values],
        columns=time)


def run(channel_data):
    channel_path = Path('data') / channel_data['name']
    channel_path.mkdir(exist_ok=True)
    wind_driven_shifts = [channel_data['wind_speed'] * t for t in channel_data['times']]
    
    tc_res = [TimeCoherenceResult(
        channel_data['channel'],
        shifts=wind_driven_shifts,
        pupil=pyatm.CirclePupil(a),
        save_path=channel_path / f"aperture_{str(a).replace('.', '_')}.raw.csv",
        max_size=channel_data['iterations'],
        ) for a in channel_data['aperture_radiuses']]

    sim = pyatm.simulations.Simulation(tc_res)
    sim.run(save_step=channel_data['save_step'])

    for a in channel_data['aperture_radiuses']:
        raw_path = channel_path / f"aperture_{str(a).replace('.', '_')}.raw.csv"
        path = channel_path / f"aperture_{str(a).replace('.', '_')}.csv"
        load_tc(raw_path, wind_driven_shifts).to_csv(path, index=False)
    

if __name__ == "__main__":
    for channel_data in CHANNELS[::-1]:
        if channel_data['name'] in []:
            continue
        run(channel_data)
