import numpy as np

from revision.config import FEATURES_24, LEGACY_18_FEATURES, LEGACY_DROPPED_FEATURES
from revision.preprocessing import FeaturePreprocessor


def test_all_24_features_and_large_values_are_retained():
    x = np.ones((4, 20, 24), dtype=float)
    x[:, :, FEATURES_24.index("TOTFX")] = 2.5e9
    x[0, 0, FEATURES_24.index("MEANGAM")] = np.nan
    processor = FeaturePreprocessor(FEATURES_24).fit(x)
    transformed = processor.transform(x)
    assert transformed.shape == (4, 20, 24)
    assert np.isfinite(transformed).all()
    audit = {row["feature"]: row for row in processor.audit}
    assert audit["TOTFX"]["raw_max"] == 2.5e9
    assert len(audit) == 24


def test_scaler_is_fit_only_on_training_values():
    train = np.zeros((2, 20, 24), dtype=float)
    test = np.full((2, 20, 24), 1e12, dtype=float)
    processor = FeaturePreprocessor(FEATURES_24).fit(train)
    before = processor.centers.copy()
    processor.transform(test)
    assert np.array_equal(before, processor.centers)


def test_legacy_18_excludes_exactly_six_declared_features():
    assert len(LEGACY_18_FEATURES) == 18
    assert set(FEATURES_24) - set(LEGACY_18_FEATURES) == set(LEGACY_DROPPED_FEATURES)
