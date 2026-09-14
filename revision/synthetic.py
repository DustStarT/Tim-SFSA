"""Small deterministic event-sample dataset for end-to-end smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES_24


def make_synthetic_event_samples(seed: int = 2345) -> list[dict]:
    rng = np.random.default_rng(seed)
    samples = []
    group_index = 0
    for partition, year, groups in ((1, 2012, 8), (2, 2013, 8), (3, 2014, 8), (4, 2015, 8), (5, 2017, 8)):
        for _ in range(groups):
            group_index += 1
            latent = rng.normal()
            event_count = 2 if latent > -0.4 else 1
            for interval in range(event_count + 1):
                prediction_start = pd.Timestamp(year=year, month=1 + group_index % 10, day=1 + interval)
                event = int(interval < event_count)
                if event:
                    duration = float(np.clip(60.0 - 18.0 * latent + rng.normal(0, 12), 2, 140))
                    event_time = prediction_start + pd.Timedelta(hours=duration)
                else:
                    duration = float(rng.uniform(150, 260))
                    event_time = None
                trend = np.linspace(-0.4, 0.4, 20)[:, None]
                loadings = np.linspace(0.2, 1.2, len(FEATURES_24))[None, :]
                features = latent * loadings + trend * loadings + rng.normal(
                    0, 0.35, size=(20, len(FEATURES_24))
                )
                # Exercise the fixed >1e6 bug without creating overflow.
                features[:, FEATURES_24.index("TOTFX")] *= 2.0e7
                features[:, FEATURES_24.index("TOTFY")] *= 3.0e7
                samples.append({
                    "sample_id": f"synthetic_harp_{group_index}__event_anchor_{interval:03d}",
                    "features": features.astype(np.float32),
                    "feature_names": list(FEATURES_24),
                    "duration": duration,
                    "duration_hours": duration,
                    "duration_units": "hours",
                    "event": event,
                    "record_id": {
                        "raw": f"ar{group_index}", "ar": f"ar{group_index}",
                        "harp_id": str(group_index), "partition": partition,
                        "prediction_start": prediction_start,
                        "event_time": event_time, "event_id": f"synthetic-{group_index}-{interval}" if event else None,
                        "event_class": "M1.0" if event else None,
                        "covered_events": [], "censor_reason": None if event else "observation_end",
                    },
                    "causal_flare_history": {
                        "B_COUNT_24H": float(rng.integers(0, 3)),
                        "C_COUNT_24H": float(rng.integers(0, 3)),
                    },
                })
    return samples
