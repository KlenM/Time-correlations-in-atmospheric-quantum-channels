from functools import partial

import numpy as np
import pandas as pd
import pyatmosphere as pyatm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


WIND_SPEED = 10
TIME = np.asarray([0, 0.0028, 0.0056, 0.0084, 0.0112, 0.014,
        0.0168, 0.0196, 0.0224, 0.0252, 0.028 ])
shifts = list(TIME * WIND_SPEED)
aperture_radiuses = [0.004, 0.006, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
pyatm.gpu.config['use_gpu'] = True


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


if __name__ == "__main__":
    channel = pyatm.Channel(
        grid=pyatm.RectGrid(resolution=2**11, delta=0.001),
            source=pyatm.GaussianSource(
            wvl=808e-9,
            w0=0.08,
            F0=50e3
            ),
        path=pyatm.IdenticalPhaseScreensPath(
            phase_screen=pyatm.SSPhaseScreen(
                model=pyatm.MVKModel(
                Cn2=2e-16,
                l0=1e-3,
                L0=80,
                ),
            f_grid=pyatm.RandLogPolarGrid(
                points=2**10,
                f_min=1 / 80 / 15,
                f_max=1 / 1e-3 * 2
                )
            ),
            length=50e3,
            count=15
            ),
        pupil=pyatm.CirclePupil(
            radius=0.2
            ),
        )

    tc_res = [TimeCoherenceResult(
        channel,
        shifts=shifts,
        pupil=pyatm.CirclePupil(a),
        save_path=f"data/raw_strong_{str(a).replace('.', '_')}.csv",
        max_size=50000,
        ) for a in aperture_radiuses]

    sim = pyatm.simulations.Simulation(tc_res)
    sim.run(save_step=1000)

    for a in aperture_radiuses:
        raw_path = f"data/raw_strong_{str(a).replace('.', '_')}.csv"
        path = f"data/strong_{str(a).replace('.', '_')}.csv"
        load_tc(raw_path, shifts).to_csv(path, index=False)
