"""Independent on-demand, finite-mode Matérn temperature source; no GP or ROS imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
import gstools as gs


class TemperatureField:
    """One fixed random realization in dimensionless (x/Lx, y/Ly, t/Lt)."""

    def __init__(self, config):
        self.config = json.loads(json.dumps(config))
        self.scales = np.asarray(config["scaled_coordinate_scales_m_m_s"], dtype=float)
        if self.scales.shape != (3,) or not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise ValueError("Three positive coordinate scales required (m, m, s)")
        self.batch_size = int(config["batch_size"])
        if self.batch_size < 1:
            raise ValueError("Positive batch size required")
        gs.config.NUM_THREADS = int(config["threads"])
        self.model = gs.Matern(spatial_dim=2, temporal=True,
            var=float(config["variance_c2"]), len_scale=1.0,
            nu=float(config["matern_nu"]), nugget=0.0)
        self.srf = gs.SRF(self.model, mean=0.0, generator="RandMeth",
            seed=int(config["field_seed"]), mode_no=int(config["fourier_modes"]))
        self.max_evaluated_batch = 0

    def _validate(self, inputs):
        z = np.asarray(inputs, dtype=float)
        if z.ndim == 1 and z.shape == (3,):
            z = z.reshape(1, 3)
        if z.ndim != 2 or z.shape[1] != 3 or not np.isfinite(z).all():
            raise ValueError("Expected finite (N,3) physical x_m, y_m, t_s")
        lower = np.array([0., 0., self.config["mission_interval_s"][0]])
        upper = np.array([*self.config["working_extent_m"], self.config["mission_interval_s"][1]])
        if np.any(z < lower) or np.any(z > upper):
            raise ValueError("Query outside declared temporary domain/time interval")
        return z

    def truth_chunks(self, inputs):
        z = self._validate(inputs)
        for start in range(0, len(z), self.batch_size):
            part = z[start:start + self.batch_size]
            self.max_evaluated_batch = max(self.max_evaluated_batch, len(part))
            fluctuation = self.srf((part / self.scales).T, mesh_type="unstructured", store=False)
            yield (float(self.config["mean_c"]) +
                   float(self.config["trend_c_per_s"]) * part[:, 2] + fluctuation)

    def truth(self, inputs):
        """Return only requested temperatures; the caller owns this output."""
        parts = list(self.truth_chunks(inputs))
        return np.concatenate(parts) if parts else np.empty(0, dtype=float)

    def noise(self, measurement_ids):
        """Stable ID-keyed noise, independent of truth queries and query order."""
        sigma = float(self.config["noise_std_c"])
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Nonnegative finite noise standard deviation required")
        values = []
        for identity in measurement_ids:
            if not isinstance(identity, str) or not identity:
                raise ValueError("Nonempty stable string measurement identity required")
            key = f'{self.config["noise_seed"]}:{identity}'.encode("utf-8")
            seed = int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "little")
            values.append(np.random.Generator(np.random.PCG64(seed)).normal(0., sigma))
        return np.asarray(values)

    def measure(self, inputs, measurement_ids):
        z = self._validate(inputs)
        if len(z) != len(measurement_ids):
            raise ValueError("One identity per measurement required")
        return self.truth(z) + self.noise(measurement_ids)

    def storage_summary(self):
        arrays = {key: list(value.shape) for key, value in vars(self.srf.generator).items()
                  if isinstance(value, np.ndarray)}
        return dict(stored_field_names=list(self.srf.field_names),
                    retained_position_count=0 if self.srf.pos is None else int(np.asarray(self.srf.pos).shape[-1]),
                    generator_array_shapes=arrays,
                    generator_array_bytes=sum(value.nbytes for value in vars(self.srf.generator).values()
                                              if isinstance(value, np.ndarray)),
                    max_evaluated_batch=self.max_evaluated_batch)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    field = TemperatureField(json.loads(Path(a.config).read_text()))
    np.save(a.output, field.truth(np.load(a.inputs)))
