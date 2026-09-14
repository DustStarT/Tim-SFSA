"""Load raw SWAN-SF HARP series and catalog-resolved flare timelines."""

from __future__ import annotations

import json
import logging
import os
import re

import pandas as pd

from . import file_manager


LOGGER = logging.getLogger(__name__)


class DataManager:
    """Raw-only loader used by the event-anchored survival pipeline."""

    def __init__(self, config):
        self.config = config
        self.specified_features = list(config.get("specified_features", []))
        self.require_flare_catalog = bool(config.get("require_flare_catalog", True))
        self.allow_raw_timestamp_event_fallback = bool(
            config.get("allow_raw_timestamp_event_fallback", False)
        )
        self.skipped_major_flare_labels = 0
        self.raw_timestamp_event_fallbacks = 0
        self.unmatched_major_events = []

    def load_and_merge_data(self):
        source_mode = str(self.config.get("source_mode", "raw_harp")).lower()
        if source_mode != "raw_harp":
            raise ValueError(
                f"Unsupported SWAN-SF source_mode={source_mode!r}. The revision "
                "accepts only raw_harp because legacy instance files cannot "
                "provide event-anchored follow-up and exact censoring times."
            )
        return self._load_raw_harp_data()

    def _load_raw_harp_data(self):
        partitions = list(self.config.get("partitions_to_process", []))
        if not partitions:
            raise ValueError("No SWAN-SF partitions configured")
        flare_catalog = self._load_flare_catalog()
        samples = []
        for partition in partitions:
            raw_files = file_manager.get_raw_harp_files(self.config["data_dir"], partition)
            LOGGER.info("Loading SWAN-SF partition %s: %s HARP files", partition, len(raw_files))
            for path in raw_files:
                sample = self._load_raw_harp_file(path, flare_catalog, partition)
                if sample is not None:
                    samples.append(sample)
        LOGGER.info("Loaded %s raw HARP parent series", len(samples))
        if self.skipped_major_flare_labels:
            LOGGER.warning(
                "Skipped %s M/X labels not found in the official flare catalog",
                self.skipped_major_flare_labels,
            )
        return samples

    def _find_flare_catalog_path(self):
        data_dir = self.config["data_dir"]
        candidates = (
            os.path.join(data_dir, "integrated_flare_data", "goes_flares_integrated.csv"),
            os.path.join(os.path.dirname(data_dir), "integrated_flare_data", "goes_flares_integrated.csv"),
        )
        return next((path for path in candidates if os.path.isfile(path)), None)

    @staticmethod
    def _first_value(row, names):
        for name in names:
            value = row.get(name)
            if value is not None and not pd.isna(value) and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _normalise_timestamp(value):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return pd.NaT
        timestamp = pd.Timestamp(parsed)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp

    @classmethod
    def _sol_identifier(cls, row):
        value = cls._first_value(
            row, ("sol_standard", "SOL_standard", "SOL_STANDARD", "sol_id", "event_id")
        )
        # ``event_id`` is ambiguous across catalogs.  Never present a numeric
        # database key as a Solar Object Locator.
        return value if re.match(r"^SOL\d{4}-\d{2}-\d{2}T", value, flags=re.IGNORECASE) else ""

    def _load_flare_catalog(self):
        path = self._find_flare_catalog_path()
        if path is None:
            message = (
                "Cannot find integrated_flare_data/goes_flares_integrated.csv; "
                "exact M/X onset times are required"
            )
            if self.require_flare_catalog:
                raise FileNotFoundError(message)
            LOGGER.warning(message)
            return {}
        frame = pd.read_csv(path)
        catalog = {}
        for _, row in frame.iterrows():
            try:
                flare_id = int(row["flare_id"])
            except (KeyError, TypeError, ValueError):
                continue
            onset = self._normalise_timestamp(row.get("start_time"))
            if pd.isna(onset):
                continue
            candidate = {
                "flare_id": flare_id,
                "flare_class": self._first_value(row, ("goes_class", "class", "magnitude")),
                "time": onset,
                "source": "goes_flares_integrated.csv",
                "sol_standard": self._sol_identifier(row),
                "noaa_active_region": self._first_value(
                    row, ("noaa_active_region", "noaa_ar", "NOAA_AR", "ar_noaanum")
                ),
                "catalog_row_count": 1,
                "catalog_duplicate_row_count": 0,
                "catalog_conflict": False,
                "catalog_alternate_records": [],
            }
            if flare_id not in catalog:
                catalog[flare_id] = candidate
                continue
            retained = catalog[flare_id]
            retained["catalog_row_count"] = int(retained.get("catalog_row_count", 1)) + 1
            retained["catalog_duplicate_row_count"] = int(
                retained.get("catalog_duplicate_row_count", 0)
            ) + 1
            conflict = (
                retained.get("time") != candidate.get("time")
                or retained.get("flare_class") != candidate.get("flare_class")
                or retained.get("sol_standard") != candidate.get("sol_standard")
            )
            retained["catalog_conflict"] = bool(retained.get("catalog_conflict")) or conflict
            retained["catalog_alternate_records"].append({
                "time": candidate["time"],
                "flare_class": candidate["flare_class"],
                "sol_standard": candidate["sol_standard"],
            })
        LOGGER.info("Loaded %s catalog flare onsets from %s", len(catalog), path)
        return catalog

    @staticmethod
    def _parse_flare_label(raw_label):
        if raw_label is None or pd.isna(raw_label):
            return {}
        text = str(raw_label).strip()
        if not text or text.lower() in {"none", "nan"}:
            return {}
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except Exception:
            event_id = re.search(r"['\"]id['\"]\s*:\s*(\d+)", text)
            magnitude = re.search(r"['\"]magnitude['\"]\s*:\s*['\"]([^'\"]+)", text)
            return {
                "id": int(event_id.group(1)) if event_id else None,
                "magnitude": magnitude.group(1) if magnitude else None,
            }

    def _extract_flare_events(
        self, raw_frame, catalog, prefixes=("M", "X"), source_file=None
    ):
        events = []
        for prefix in prefixes:
            value_column = f"{prefix}FLARE"
            label_column = f"{prefix}FLARE_LABEL"
            if value_column not in raw_frame.columns and label_column not in raw_frame.columns:
                continue
            values = pd.to_numeric(
                raw_frame.get(value_column, pd.Series(0, index=raw_frame.index)), errors="coerce"
            ).fillna(0)
            labels = raw_frame.get(
                label_column, pd.Series(["None"] * len(raw_frame), index=raw_frame.index)
            )
            present_label = ~labels.astype(str).str.strip().str.lower().isin({"", "none", "nan"})
            for index in raw_frame.index[(values > 0) | present_label]:
                parsed = self._parse_flare_label(labels.loc[index])
                try:
                    flare_id = int(parsed.get("id"))
                except (TypeError, ValueError):
                    flare_id = None
                event = dict(catalog.get(flare_id, {}))
                if not event:
                    if not self.allow_raw_timestamp_event_fallback:
                        if prefix in {"M", "X"}:
                            self.skipped_major_flare_labels += 1
                            self.unmatched_major_events.append({
                                "label_prefix": prefix,
                                "flare_id": flare_id,
                                "raw_label": str(labels.loc[index]),
                                "raw_timestamp": raw_frame.loc[index, "timestamp_dt"],
                                "source_file": source_file,
                            })
                        continue
                    self.raw_timestamp_event_fallbacks += 1
                    event = {
                        "flare_id": flare_id,
                        "flare_class": str(parsed.get("magnitude") or prefix),
                        "time": raw_frame.loc[index, "timestamp_dt"],
                        "source": "raw_harp_timestamp_fallback",
                        "sol_standard": "",
                        "noaa_active_region": "",
                    }
                event.update({
                    "label_prefix": prefix,
                    "raw_label": str(labels.loc[index]),
                    "raw_label_timestamp": raw_frame.loc[index, "timestamp_dt"],
                    "catalog_match": event.get("source") == "goes_flares_integrated.csv",
                    "catalog_time_offset_hours": (
                        event["time"] - raw_frame.loc[index, "timestamp_dt"]
                    ).total_seconds() / 3600.0,
                    "raw_labels": [str(labels.loc[index])],
                    "source_occurrence_count": 1,
                    "duplicate_occurrence_count": 0,
                })
                events.append(event)
        deduplicated = {}
        for event in events:
            event_time = self._normalise_timestamp(event.get("time"))
            if pd.isna(event_time):
                continue
            event["time"] = event_time
            event_id = event.get("flare_id")
            key = ("id", event_id) if event_id is not None else (
                event.get("flare_class"), event["time"].isoformat()
            )
            if key not in deduplicated:
                deduplicated[key] = event
                continue
            retained = deduplicated[key]
            retained["source_occurrence_count"] = int(
                retained.get("source_occurrence_count", 1)
            ) + 1
            retained["duplicate_occurrence_count"] = int(
                retained.get("duplicate_occurrence_count", 0)
            ) + 1
            retained_labels = list(retained.get("raw_labels", []))
            candidate_label = str(event.get("raw_label", ""))
            if candidate_label and candidate_label not in retained_labels:
                retained_labels.append(candidate_label)
            retained["raw_labels"] = retained_labels
        return sorted(deduplicated.values(), key=lambda item: item["time"])

    @staticmethod
    def _timestamp_column(frame):
        for name in ("timestamp", "Timestamp", "TIMESTAMP"):
            if name in frame.columns:
                return name
        return None

    def _load_raw_harp_file(self, path, catalog, partition):
        try:
            raw = pd.read_csv(path, sep="\t", engine="python")
        except Exception as exc:
            LOGGER.warning("Cannot read %s: %s", path, exc)
            return None
        timestamp_column = self._timestamp_column(raw)
        if timestamp_column is None:
            LOGGER.warning("Skipping raw HARP file without a timestamp column: %s", path)
            return None
        raw["timestamp_dt"] = raw[timestamp_column].map(self._normalise_timestamp)
        raw.dropna(subset=["timestamp_dt"], inplace=True)
        raw.sort_values("timestamp_dt", inplace=True)
        raw.drop_duplicates(subset=["timestamp_dt"], keep="first", inplace=True)
        raw.reset_index(drop=True, inplace=True)
        if raw.empty:
            return None

        # Missing and malformed values remain NaN until training-only imputation.
        features = raw.reindex(columns=self.specified_features).apply(pd.to_numeric, errors="coerce")
        flare_history = self._extract_flare_events(
            raw, catalog, prefixes=("B", "C", "M", "X"), source_file=path
        )
        major_events = [
            event for event in flare_history
            if str(event.get("label_prefix", "")).upper() in {"M", "X"}
        ]
        start = pd.Timestamp(raw["timestamp_dt"].iloc[0])
        end = pd.Timestamp(raw["timestamp_dt"].iloc[-1])
        harp_id = os.path.splitext(os.path.basename(path))[0]
        ar_id = None
        for candidate in ("NOAA_AR", "NOAA_ARS", "NOAAAR", "AR"):
            if candidate not in raw.columns:
                continue
            values = raw[candidate].dropna().astype(str).str.strip()
            values = values[~values.isin({"", "0", "nan", "None"})]
            if not values.empty:
                ar_id = values.iloc[-1]
                break
        first_event = next((item for item in major_events if start <= item["time"] <= end), None)
        followup_end = first_event["time"] if first_event is not None else end
        duration = max(0.0, (followup_end - start).total_seconds() / 3600.0)
        return {
            "record_id": f"ar{harp_id}", "harp_id": harp_id, "ar_id": ar_id,
            "partition": int(partition), "features": features,
            "timestamps_list": raw[timestamp_column].astype(str).tolist(),
            "major_events": major_events, "flare_history_events": flare_history,
            "event": int(bool(major_events)), "time": duration, "duration": duration,
            "duration_hours": duration, "duration_units": "hours",
            "event_time_raw": first_event["time"] if first_event else None,
            "start_time": start, "end_time": end, "source_file": path,
        }
