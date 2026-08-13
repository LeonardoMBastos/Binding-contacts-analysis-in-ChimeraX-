"""Performance timing extraction and plotting for DockAnalyzer.

The module is intentionally independent from the scientific analysis pipeline.
It reads timing information already produced by DockAnalyzer and creates two
optional diagnostic figures:

- ``execution_timeline.png``: reconstructed per-pose stage timeline;
- ``stage_runtime.png``: accumulated runtime by analysis stage.

The structured ``*_run_summary.json`` file is the preferred input because it
preserves pose identifiers, ranks, stage durations, parallel wall time, and
runtime accounting.  A DockAnalyzer ``.log`` file is also supported as a
fallback.  Log-derived pose identities are matched through the final
interaction counts when that mapping is unambiguous.

No scientific scores, interaction criteria, rankings, or analysis results are
modified by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from ._version import __version__
except Exception:  # pragma: no cover - supports direct standalone execution
    __version__ = "unknown"


PathLike = Union[str, Path]
PerformanceSource = Union[PathLike, Mapping[str, Any]]

PERFORMANCE_SCHEMA_NAME = "dockanalyzer.performance"
PERFORMANCE_SCHEMA_VERSION = "1.0"
DEFAULT_TIMELINE_FILENAME = "execution_timeline.png"
DEFAULT_RUNTIME_FILENAME = "stage_runtime.png"
DEFAULT_IMAGE_WIDTH = 1800
DEFAULT_TIMELINE_ROW_HEIGHT = 64
DEFAULT_RUNTIME_ROW_HEIGHT = 54
DEFAULT_IMAGE_MARGIN = 80
DEFAULT_MIN_BAR_SECONDS = 0.02
DEFAULT_STAGE_ORDER = (
    "input_resolution",
    "model_preparation",
    "contacts",
    "hbonds",
    "hydrophobic",
    "pi",
    "saltbridge",
    "consolidation",
    "scoring",
    "consensus",
    "validation",
    "export",
    "report",
    "visualization",
    "finalization",
)

_LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|"
    r"\s*(?P<level>[^|]+)\|\s*(?P<module>[^|]+)\|\s*(?P<message>.*)$"
)
_STAGE_FINISHED_RE = re.compile(
    r"^Stage finished:\s*(?P<stage>[^;]+);\s*status=(?P<status>[^;]+);"
    r"\s*duration_seconds=(?P<duration>[0-9eE+\-.]+)(?P<rest>.*)$"
)
_SELECTED_POSE_RE = re.compile(
    r"^Selected pose\s+\d+:\s*name=(?P<name>.*?),\s*identifier=(?P<pose_id>[^,]+),"
)
_POSE_RESULT_RE = re.compile(
    r"^Pose result:\s*pose_id=(?P<pose_id>[^,]+),\s*name=(?P<name>.*?),.*?"
    r"interactions=(?P<interactions>\d+)"
)
_KEY_VALUE_RE = re.compile(r"(?:^|;)\s*([A-Za-z0-9_]+)=([^;]*)")


@dataclass(frozen=True)
class StageTiming:
    """One measured stage duration."""

    stage: str
    duration_seconds: float
    scope: str = "pose"
    pose_id: Optional[str] = None
    pose_name: Optional[str] = None
    rank: Optional[int] = None
    status: str = "unknown"
    order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        duration = _finite_nonnegative(self.duration_seconds)
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "stage", str(self.stage).strip() or "unknown")
        object.__setattr__(self, "scope", str(self.scope).strip() or "pose")
        if self.pose_id is not None:
            object.__setattr__(self, "pose_id", str(self.pose_id))
        if self.pose_name is not None:
            object.__setattr__(self, "pose_name", str(self.pose_name))
        if self.rank is not None:
            try:
                object.__setattr__(self, "rank", int(self.rank))
            except (TypeError, ValueError):
                object.__setattr__(self, "rank", None)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready timing record."""

        return {
            "stage": self.stage,
            "duration_seconds": self.duration_seconds,
            "scope": self.scope,
            "pose_id": self.pose_id,
            "pose_name": self.pose_name,
            "rank": self.rank,
            "status": self.status,
            "order": self.order,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TimelineSegment:
    """One reconstructed interval used by the timeline plot."""

    label: str
    stage: str
    start_seconds: float
    duration_seconds: float
    pose_id: Optional[str] = None
    rank: Optional[int] = None
    scope: str = "pose"

    @property
    def end_seconds(self) -> float:
        """Return interval end time."""

        return self.start_seconds + self.duration_seconds


@dataclass(frozen=True)
class PerformanceData:
    """Normalized performance data from one DockAnalyzer run."""

    source: Optional[Path] = None
    source_type: str = "mapping"
    pose_timings: Tuple[StageTiming, ...] = ()
    global_timings: Tuple[StageTiming, ...] = ()
    planned_order: Tuple[str, ...] = DEFAULT_STAGE_ORDER
    total_duration_seconds: Optional[float] = None
    pose_runtime_seconds: Optional[float] = None
    pose_wall_runtime_seconds: Optional[float] = None
    parallel_overlap_seconds: Optional[float] = None
    worker_count: Optional[int] = None
    parallel_enabled: Optional[bool] = None
    warnings: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "pose_timings", tuple(self.pose_timings))
        object.__setattr__(self, "global_timings", tuple(self.global_timings))
        object.__setattr__(self, "planned_order", tuple(self.planned_order or DEFAULT_STAGE_ORDER))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    @property
    def pose_ids(self) -> Tuple[str, ...]:
        """Return pose identifiers in rank/input order."""

        ordered: List[Tuple[int, int, str]] = []
        seen = set()
        for index, item in enumerate(self.pose_timings):
            if item.pose_id is None or item.pose_id in seen:
                continue
            seen.add(item.pose_id)
            rank_key = item.rank if item.rank is not None else 10**9
            ordered.append((rank_key, index, item.pose_id))
        return tuple(item[2] for item in sorted(ordered))

    def summary(self) -> Dict[str, Any]:
        """Return a compact data summary."""

        return {
            "schema": PERFORMANCE_SCHEMA_NAME,
            "schema_version": PERFORMANCE_SCHEMA_VERSION,
            "source": str(self.source) if self.source else None,
            "source_type": self.source_type,
            "pose_count": len(self.pose_ids),
            "pose_stage_count": len(self.pose_timings),
            "global_stage_count": len(self.global_timings),
            "total_duration_seconds": self.total_duration_seconds,
            "pose_runtime_seconds": self.pose_runtime_seconds,
            "pose_wall_runtime_seconds": self.pose_wall_runtime_seconds,
            "parallel_overlap_seconds": self.parallel_overlap_seconds,
            "worker_count": self.worker_count,
            "parallel_enabled": self.parallel_enabled,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PerformancePlotResult:
    """Paths and metadata returned after figure generation."""

    timeline_path: Path
    runtime_path: Path
    backend: str
    data: PerformanceData
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready plot result."""

        return {
            "schema": PERFORMANCE_SCHEMA_NAME,
            "schema_version": PERFORMANCE_SCHEMA_VERSION,
            "timeline_path": str(self.timeline_path),
            "runtime_path": str(self.runtime_path),
            "backend": self.backend,
            "data": self.data.summary(),
            "warnings": list(self.warnings),
        }


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    """Return a finite non-negative float."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return max(0.0, number)


def _optional_float(value: Any) -> Optional[float]:
    """Return a finite float or ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> Optional[int]:
    """Return an integer or ``None``."""

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping view for supported values."""

    return value if isinstance(value, Mapping) else {}


def _stage_order_index(stage: str, planned_order: Sequence[str]) -> int:
    """Return a stable order index for one stage."""

    try:
        return tuple(planned_order).index(stage)
    except ValueError:
        return len(tuple(planned_order)) + 100


def _rank_metadata(summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return pose rank/name metadata keyed by pose id."""

    result: Dict[str, Dict[str, Any]] = {}
    ranking = _mapping(summary.get("ranking"))
    entries = ranking.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        return result
    for entry in entries:
        record = _mapping(entry)
        pose_id = record.get("pose_id")
        if pose_id in (None, ""):
            continue
        result[str(pose_id)] = {
            "rank": _optional_int(record.get("rank") or record.get("position")),
            "name": record.get("name"),
            "score": _optional_float(record.get("score")),
            "status": record.get("status"),
        }
    return result


def extract_run_summary_timings(
    summary: Mapping[str, Any],
    *,
    source: Optional[PathLike] = None,
) -> PerformanceData:
    """Extract structured performance data from a DockAnalyzer run summary."""

    stages = _mapping(summary.get("stages"))
    planned_raw = stages.get("planned_order")
    planned_order = tuple(
        str(item) for item in planned_raw
    ) if isinstance(planned_raw, Sequence) and not isinstance(planned_raw, (str, bytes, bytearray)) else DEFAULT_STAGE_ORDER
    rank_by_pose = _rank_metadata(summary)

    pose_records: List[StageTiming] = []
    poses = _mapping(stages.get("poses"))
    for pose_index, (pose_id_raw, stage_map_raw) in enumerate(poses.items()):
        pose_id = str(pose_id_raw)
        rank_info = rank_by_pose.get(pose_id, {})
        stage_map = _mapping(stage_map_raw)
        for local_index, (stage_raw, values_raw) in enumerate(stage_map.items()):
            stage = str(stage_raw)
            values = _mapping(values_raw)
            pose_records.append(
                StageTiming(
                    stage=stage,
                    duration_seconds=values.get("duration_seconds", 0.0),
                    scope="pose",
                    pose_id=pose_id,
                    pose_name=rank_info.get("name"),
                    rank=rank_info.get("rank"),
                    status=str(values.get("status", "unknown")),
                    order=_stage_order_index(stage, planned_order) * 1000 + local_index,
                    metadata={"pose_index": pose_index},
                )
            )

    global_records: List[StageTiming] = []
    global_map = _mapping(stages.get("global"))
    for local_index, (stage_raw, values_raw) in enumerate(global_map.items()):
        stage = str(stage_raw)
        values = _mapping(values_raw)
        global_records.append(
            StageTiming(
                stage=stage,
                duration_seconds=values.get("duration_seconds", 0.0),
                scope="global",
                status=str(values.get("status", "unknown")),
                order=_stage_order_index(stage, planned_order) * 1000 + local_index,
            )
        )

    timings = _mapping(summary.get("timings"))
    parallel = _mapping(summary.get("parallel_execution"))
    warnings: List[str] = []
    if not pose_records:
        warnings.append("The run summary does not contain per-pose stage timings.")
    if not global_records:
        warnings.append("The run summary does not contain global stage timings.")

    return PerformanceData(
        source=Path(source) if source is not None else None,
        source_type="run_summary",
        pose_timings=tuple(pose_records),
        global_timings=tuple(global_records),
        planned_order=planned_order,
        total_duration_seconds=_optional_float(
            timings.get("duration_seconds") or timings.get("analysis_duration_seconds")
        ),
        pose_runtime_seconds=_optional_float(timings.get("pose_runtime_seconds")),
        pose_wall_runtime_seconds=_optional_float(
            timings.get("pose_wall_runtime_seconds") or parallel.get("duration_seconds")
        ),
        parallel_overlap_seconds=_optional_float(timings.get("parallel_overlap_seconds")),
        worker_count=_optional_int(parallel.get("worker_count")),
        parallel_enabled=(bool(parallel.get("enabled")) if "enabled" in parallel else None),
        warnings=tuple(warnings),
        metadata={
            "run_id": summary.get("run_id"),
            "request_id": summary.get("request_id"),
            "status": summary.get("status"),
            "ranking": rank_by_pose,
            "timings": dict(timings),
            "parallel_execution": dict(parallel),
        },
    )


def _parse_log_record(line: str) -> Optional[Tuple[datetime, str]]:
    """Return timestamp and message for one DockAnalyzer log line."""

    match = _LOG_LINE_RE.match(line.rstrip("\r\n"))
    if not match:
        return None
    try:
        timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return timestamp, match.group("message").strip()


def _parse_stage_message(message: str) -> Optional[Dict[str, Any]]:
    """Parse a ``Stage finished`` log message."""

    match = _STAGE_FINISHED_RE.match(message)
    if not match:
        return None
    record: Dict[str, Any] = {
        "stage": match.group("stage").strip(),
        "status": match.group("status").strip(),
        "duration_seconds": _finite_nonnegative(match.group("duration")),
    }
    rest = match.group("rest")
    for key, value in _KEY_VALUE_RE.findall(rest):
        record[key] = value.strip()
    return record


def parse_log_timings(path: PathLike) -> PerformanceData:
    """Extract performance data from a DockAnalyzer text log.

    Worker stage callbacks can be flushed in blocks after a pose completes, so
    log timestamps are not treated as exact per-stage wall-clock boundaries.
    The timeline is therefore reconstructed from each recorded stage duration.
    """

    source = Path(path)
    selected_poses: Dict[str, str] = {}
    pose_result_by_interactions: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    all_stage_events: List[Tuple[datetime, Dict[str, Any]]] = []
    timestamps: List[datetime] = []

    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        parsed = _parse_log_record(line)
        if parsed is None:
            continue
        timestamp, message = parsed
        timestamps.append(timestamp)
        selected_match = _SELECTED_POSE_RE.match(message)
        if selected_match:
            selected_poses[selected_match.group("pose_id").strip()] = selected_match.group("name").strip()
        result_match = _POSE_RESULT_RE.match(message)
        if result_match:
            pose_result_by_interactions[int(result_match.group("interactions"))].append(
                {
                    "pose_id": result_match.group("pose_id").strip(),
                    "name": result_match.group("name").strip(),
                }
            )
        stage = _parse_stage_message(message)
        if stage is not None:
            all_stage_events.append((timestamp, stage))

    # The first global input/model-preparation stages occur before worker blocks.
    # Worker blocks end with a pose-level finalization containing interaction_count.
    pose_blocks: List[List[Dict[str, Any]]] = []
    global_events: List[Dict[str, Any]] = []
    current_block: List[Dict[str, Any]] = []
    worker_region_started = False
    consensus_seen = False

    for _, stage in all_stage_events:
        stage_name = str(stage.get("stage", ""))
        is_pose_finalization = stage_name == "finalization" and "interaction_count" in stage
        if stage_name == "consensus":
            if current_block:
                pose_blocks.append(current_block)
                current_block = []
            consensus_seen = True
            global_events.append(stage)
            continue
        if consensus_seen:
            global_events.append(stage)
            continue
        if not worker_region_started:
            if stage_name == "input_resolution" and "item_count" in stage:
                worker_region_started = True
                current_block = [stage]
            else:
                global_events.append(stage)
            continue
        current_block.append(stage)
        if is_pose_finalization:
            pose_blocks.append(current_block)
            current_block = []
    if current_block:
        pose_blocks.append(current_block)

    warnings: List[str] = []
    pose_records: List[StageTiming] = []
    used_pose_ids = set()
    for block_index, block in enumerate(pose_blocks, start=1):
        finalization = next(
            (item for item in reversed(block) if item.get("stage") == "finalization"),
            {},
        )
        interaction_count = _optional_int(finalization.get("interaction_count"))
        matches = pose_result_by_interactions.get(interaction_count or -1, [])
        available_matches = [item for item in matches if item.get("pose_id") not in used_pose_ids]
        if len(available_matches) == 1:
            identity = available_matches[0]
            pose_id = str(identity.get("pose_id"))
            pose_name = str(identity.get("name") or selected_poses.get(pose_id) or "") or None
            used_pose_ids.add(pose_id)
        else:
            pose_id = f"worker_{block_index}"
            pose_name = None
            if interaction_count is not None:
                warnings.append(
                    "Could not unambiguously map a log worker block with "
                    f"{interaction_count} interactions to a pose id."
                )
        for local_index, values in enumerate(block):
            stage = str(values.get("stage", "unknown"))
            pose_records.append(
                StageTiming(
                    stage=stage,
                    duration_seconds=values.get("duration_seconds", 0.0),
                    scope="pose",
                    pose_id=pose_id,
                    pose_name=pose_name,
                    status=str(values.get("status", "unknown")),
                    order=_stage_order_index(stage, DEFAULT_STAGE_ORDER) * 1000 + local_index,
                    metadata={
                        "interaction_count": interaction_count,
                        "log_worker_block": block_index,
                    },
                )
            )

    global_records = tuple(
        StageTiming(
            stage=str(values.get("stage", "unknown")),
            duration_seconds=values.get("duration_seconds", 0.0),
            scope="global",
            status=str(values.get("status", "unknown")),
            order=_stage_order_index(str(values.get("stage", "unknown")), DEFAULT_STAGE_ORDER) * 1000 + index,
        )
        for index, values in enumerate(global_events)
    )

    total_duration = None
    if timestamps:
        total_duration = max(0.0, (max(timestamps) - min(timestamps)).total_seconds())
    pose_runtime = sum(item.duration_seconds for item in pose_records)
    pose_totals: Dict[str, float] = defaultdict(float)
    for item in pose_records:
        if item.pose_id is not None:
            pose_totals[item.pose_id] += item.duration_seconds
    pose_wall = max(pose_totals.values(), default=None)
    if pose_wall is not None and pose_runtime >= pose_wall:
        overlap = pose_runtime - pose_wall
    else:
        overlap = None

    if not pose_records:
        warnings.append("No per-pose stage timing blocks were found in the log.")

    return PerformanceData(
        source=source,
        source_type="log",
        pose_timings=tuple(pose_records),
        global_timings=global_records,
        planned_order=DEFAULT_STAGE_ORDER,
        total_duration_seconds=total_duration,
        pose_runtime_seconds=pose_runtime or None,
        pose_wall_runtime_seconds=pose_wall,
        parallel_overlap_seconds=overlap,
        warnings=tuple(warnings),
        metadata={
            "selected_poses": selected_poses,
            "log_line_count": len(lines),
            "timeline_reconstructed": True,
        },
    )


def load_performance_data(source: PerformanceSource) -> PerformanceData:
    """Load performance data from a mapping, run-summary JSON, or text log."""

    if isinstance(source, Mapping):
        return extract_run_summary_timings(source)
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("Performance JSON input must contain an object at the top level.")
        return extract_run_summary_timings(payload, source=path)
    if suffix in {".log", ".txt"}:
        return parse_log_timings(path)
    raise ValueError(f"Unsupported performance input format: {path.suffix or '<none>'}.")


def _pose_sort_key(item: StageTiming) -> Tuple[int, str]:
    """Return stable rank-first ordering for pose timing records."""

    return (item.rank if item.rank is not None else 10**9, item.pose_id or "")


def reconstruct_pose_timeline(
    data: PerformanceData,
    *,
    include_zero_duration: bool = False,
) -> Tuple[TimelineSegment, ...]:
    """Reconstruct sequential worker timelines from measured stage durations."""

    grouped: Dict[str, List[StageTiming]] = defaultdict(list)
    for item in data.pose_timings:
        grouped[item.pose_id or "unknown"].append(item)
    pose_order = sorted(
        grouped,
        key=lambda pose_id: _pose_sort_key(grouped[pose_id][0]),
    )
    segments: List[TimelineSegment] = []
    for pose_id in pose_order:
        timings = sorted(grouped[pose_id], key=lambda item: (item.order, item.stage))
        if not timings:
            continue
        first = timings[0]
        rank_prefix = f"Rank {first.rank} · " if first.rank is not None else ""
        label = f"{rank_prefix}Pose #{pose_id}"
        cursor = 0.0
        for item in timings:
            duration = item.duration_seconds
            if include_zero_duration or duration > 0.0:
                segments.append(
                    TimelineSegment(
                        label=label,
                        stage=item.stage,
                        start_seconds=cursor,
                        duration_seconds=duration,
                        pose_id=pose_id,
                        rank=item.rank,
                        scope="pose",
                    )
                )
            cursor += duration
    return tuple(segments)


def aggregate_stage_runtime(
    data: PerformanceData,
    *,
    include_global: bool = True,
    include_zero_duration: bool = False,
) -> Tuple[Tuple[str, float, str], ...]:
    """Return accumulated runtime by stage and scope."""

    pose_totals: Dict[str, float] = defaultdict(float)
    global_totals: Dict[str, float] = defaultdict(float)
    for item in data.pose_timings:
        pose_totals[item.stage] += item.duration_seconds
    if include_global:
        for item in data.global_timings:
            global_totals[item.stage] += item.duration_seconds

    records: List[Tuple[str, float, str]] = []
    for stage, duration in pose_totals.items():
        if include_zero_duration or duration > 0.0:
            records.append((stage, duration, "pose"))
    for stage, duration in global_totals.items():
        if include_zero_duration or duration > 0.0:
            # Keep global stages distinct when the same stage also exists per pose.
            label = f"{stage} (global)" if stage in pose_totals else stage
            records.append((label, duration, "global"))
    records.sort(key=lambda item: (-item[1], item[0]))
    return tuple(records)


def _human_seconds(value: float) -> str:
    """Format seconds compactly for chart labels."""

    if value >= 120.0:
        return f"{value / 60.0:.1f} min"
    if value >= 10.0:
        return f"{value:.1f} s"
    if value >= 1.0:
        return f"{value:.2f} s"
    return f"{value:.3f} s"


def _stage_display(stage: str) -> str:
    """Return a readable stage label."""

    return stage.replace("_", " ").strip().title()


def _stage_color(stage: str, *, alpha: int = 235) -> Tuple[int, int, int, int]:
    """Return a deterministic color for one stage."""

    palette = (
        (72, 112, 176),
        (221, 132, 82),
        (88, 153, 105),
        (196, 78, 82),
        (129, 114, 179),
        (147, 120, 96),
        (218, 139, 195),
        (140, 140, 140),
        (204, 185, 116),
        (100, 181, 205),
        (75, 135, 185),
        (181, 105, 79),
    )
    value = sum((index + 1) * ord(char) for index, char in enumerate(stage))
    red, green, blue = palette[value % len(palette)]
    return red, green, blue, alpha


def _load_pillow() -> Tuple[Any, Any, Any]:
    """Load Pillow lazily so timing extraction remains dependency-light."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Pillow is required to render DockAnalyzer performance PNG files."
        ) from exc
    return Image, ImageDraw, ImageFont


def _font(ImageFont: Any, size: int, *, bold: bool = False) -> Any:
    """Return an available sans-serif font."""

    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_axes(
    draw: Any,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    max_seconds: float,
    font: Any,
    tick_count: int = 6,
) -> None:
    """Draw a simple time axis and vertical grid lines."""

    axis_color = (65, 65, 65, 255)
    grid_color = (220, 220, 220, 255)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
    span = max(1.0, max_seconds)
    for index in range(tick_count + 1):
        fraction = index / float(tick_count)
        x = int(round(left + fraction * (right - left)))
        value = fraction * span
        draw.line((x, top, x, bottom), fill=grid_color, width=1)
        label = _human_seconds(value)
        bbox = draw.textbbox((0, 0), label, font=font)
        width = bbox[2] - bbox[0]
        draw.text((x - width / 2, bottom + 10), label, fill=axis_color, font=font)


def render_execution_timeline(
    data: PerformanceData,
    output_path: PathLike,
    *,
    width: int = DEFAULT_IMAGE_WIDTH,
    include_zero_duration: bool = False,
) -> Path:
    """Render the reconstructed per-pose execution timeline as PNG."""

    Image, ImageDraw, ImageFont = _load_pillow()
    segments = reconstruct_pose_timeline(data, include_zero_duration=include_zero_duration)
    labels: List[str] = []
    for segment in segments:
        if segment.label not in labels:
            labels.append(segment.label)
    if not labels:
        raise ValueError("No per-pose timing data are available for the execution timeline.")

    title_font = _font(ImageFont, 38, bold=True)
    subtitle_font = _font(ImageFont, 22)
    label_font = _font(ImageFont, 24, bold=True)
    text_font = _font(ImageFont, 20)
    small_font = _font(ImageFont, 18)
    margin = DEFAULT_IMAGE_MARGIN
    left = 340
    right = max(left + 500, int(width) - margin)
    top = 175
    row_height = DEFAULT_TIMELINE_ROW_HEIGHT
    bottom = top + len(labels) * row_height + 30
    height = bottom + 155
    image = Image.new("RGBA", (int(width), int(height)), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    title = "DockAnalyzer execution timeline"
    subtitle = "Reconstructed from recorded stage durations; zero-length stages are omitted."
    draw.text((margin, 35), title, fill=(25, 25, 25, 255), font=title_font)
    draw.text((margin, 92), subtitle, fill=(85, 85, 85, 255), font=subtitle_font)

    max_end = max(segment.end_seconds for segment in segments)
    if data.pose_wall_runtime_seconds is not None:
        max_end = max(max_end, data.pose_wall_runtime_seconds)
    max_end = max(max_end, 1.0)
    _draw_axes(
        draw,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        max_seconds=max_end,
        font=small_font,
    )

    segments_by_label: Dict[str, List[TimelineSegment]] = defaultdict(list)
    for segment in segments:
        segments_by_label[segment.label].append(segment)

    for row_index, label in enumerate(labels):
        y_center = top + row_index * row_height + row_height // 2
        draw.text((margin, y_center - 15), label, fill=(35, 35, 35, 255), font=label_font)
        for segment in segments_by_label[label]:
            x0 = left + int(round((segment.start_seconds / max_end) * (right - left)))
            effective_duration = max(segment.duration_seconds, DEFAULT_MIN_BAR_SECONDS)
            x1 = left + int(round(((segment.start_seconds + effective_duration) / max_end) * (right - left)))
            x1 = max(x0 + 2, x1)
            y0 = y_center - 17
            y1 = y_center + 17
            draw.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=5,
                fill=_stage_color(segment.stage),
                outline=(255, 255, 255, 255),
                width=1,
            )
            if x1 - x0 >= 80:
                stage_label = _stage_display(segment.stage)
                draw.text((x0 + 7, y0 + 5), stage_label, fill=(255, 255, 255, 255), font=text_font)

    if data.pose_wall_runtime_seconds is not None and data.pose_wall_runtime_seconds <= max_end * 1.02:
        wall_x = left + int(round((data.pose_wall_runtime_seconds / max_end) * (right - left)))
        draw.line((wall_x, top - 8, wall_x, bottom), fill=(55, 55, 55, 255), width=3)
        wall_label = f"Pose wall time: {_human_seconds(data.pose_wall_runtime_seconds)}"
        bbox = draw.textbbox((0, 0), wall_label, font=small_font)
        label_width = bbox[2] - bbox[0]
        draw.text(
            (min(right - label_width, max(left, wall_x - label_width - 8)), top - 38),
            wall_label,
            fill=(55, 55, 55, 255),
            font=small_font,
        )

    legend_stages: List[str] = []
    for segment in segments:
        if segment.stage not in legend_stages:
            legend_stages.append(segment.stage)
    legend_x = margin
    legend_y = bottom + 62
    for stage in legend_stages:
        label = _stage_display(stage)
        bbox = draw.textbbox((0, 0), label, font=small_font)
        item_width = 28 + (bbox[2] - bbox[0]) + 28
        if legend_x + item_width > width - margin:
            legend_x = margin
            legend_y += 32
        draw.rounded_rectangle(
            (legend_x, legend_y, legend_x + 18, legend_y + 18),
            radius=3,
            fill=_stage_color(stage),
        )
        draw.text((legend_x + 26, legend_y - 2), label, fill=(55, 55, 55, 255), font=small_font)
        legend_x += item_width

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, format="PNG")
    return output


def render_stage_runtime(
    data: PerformanceData,
    output_path: PathLike,
    *,
    width: int = DEFAULT_IMAGE_WIDTH,
    include_global: bool = True,
) -> Path:
    """Render accumulated stage runtime as a horizontal bar chart PNG."""

    Image, ImageDraw, ImageFont = _load_pillow()
    records = aggregate_stage_runtime(data, include_global=include_global)
    if not records:
        raise ValueError("No positive stage durations are available for the runtime chart.")

    title_font = _font(ImageFont, 38, bold=True)
    subtitle_font = _font(ImageFont, 22)
    label_font = _font(ImageFont, 22, bold=True)
    value_font = _font(ImageFont, 20)
    tick_font = _font(ImageFont, 18)
    margin = DEFAULT_IMAGE_MARGIN
    left = 520
    right = max(left + 500, int(width) - 150)
    top = 175
    row_height = DEFAULT_RUNTIME_ROW_HEIGHT
    bottom = top + len(records) * row_height + 20
    height = bottom + 115
    image = Image.new("RGBA", (int(width), int(height)), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.text((margin, 35), "DockAnalyzer runtime by stage", fill=(25, 25, 25, 255), font=title_font)
    draw.text(
        (margin, 92),
        "Pose-stage bars are accumulated across poses; global stages are shown separately.",
        fill=(85, 85, 85, 255),
        font=subtitle_font,
    )

    max_duration = max(item[1] for item in records)
    max_duration = max(max_duration, 1.0)
    _draw_axes(
        draw,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        max_seconds=max_duration,
        font=tick_font,
    )

    for index, (stage, duration, scope) in enumerate(records):
        y_center = top + index * row_height + row_height // 2
        display = _stage_display(stage)
        draw.text((margin, y_center - 14), display, fill=(35, 35, 35, 255), font=label_font)
        x1 = left + int(round((duration / max_duration) * (right - left)))
        x1 = max(left + 3, x1)
        color = _stage_color(stage, alpha=235 if scope == "pose" else 165)
        draw.rounded_rectangle(
            (left, y_center - 15, x1, y_center + 15),
            radius=5,
            fill=color,
        )
        value = _human_seconds(duration)
        draw.text((x1 + 10, y_center - 12), value, fill=(45, 45, 45, 255), font=value_font)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, format="PNG")
    return output


def generate_performance_plots(
    source: PerformanceSource,
    *,
    output_directory: Optional[PathLike] = None,
    timeline_filename: str = DEFAULT_TIMELINE_FILENAME,
    runtime_filename: str = DEFAULT_RUNTIME_FILENAME,
    include_global_runtime: bool = True,
    image_width: int = DEFAULT_IMAGE_WIDTH,
) -> PerformancePlotResult:
    """Load one run and generate both DockAnalyzer performance figures."""

    data = load_performance_data(source)
    if output_directory is None:
        if data.source is not None:
            source_parent = data.source.parent
            if source_parent.name.casefold() in {"json", "logs"}:
                output_dir = source_parent.parent / "Images"
            else:
                output_dir = source_parent / "Images"
        else:
            output_dir = Path("DockAnalyzer_Output") / "Images"
    else:
        output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = render_execution_timeline(
        data,
        output_dir / timeline_filename,
        width=image_width,
    )
    runtime_path = render_stage_runtime(
        data,
        output_dir / runtime_filename,
        width=image_width,
        include_global=include_global_runtime,
    )
    return PerformancePlotResult(
        timeline_path=timeline_path,
        runtime_path=runtime_path,
        backend="pillow",
        data=data,
        warnings=data.warnings,
    )


def performance_data_to_json(data: PerformanceData, *, indent: int = 2) -> str:
    """Serialize normalized performance data without image generation."""

    payload = data.summary()
    payload["pose_timings"] = [item.to_dict() for item in data.pose_timings]
    payload["global_timings"] = [item.to_dict() for item in data.global_timings]
    payload["planned_order"] = list(data.planned_order)
    payload["metadata"] = dict(data.metadata)
    return json.dumps(payload, indent=indent, ensure_ascii=False, default=str)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the standalone command-line parser."""

    parser = argparse.ArgumentParser(
        description="Generate DockAnalyzer performance plots from a run summary or log."
    )
    parser.add_argument("source", help="Path to *_run_summary.json or a DockAnalyzer .log file.")
    parser.add_argument(
        "--output-directory",
        default=None,
        help="Destination directory. Defaults to the run's Images directory.",
    )
    parser.add_argument(
        "--timeline-filename",
        default=DEFAULT_TIMELINE_FILENAME,
        help="Timeline PNG filename.",
    )
    parser.add_argument(
        "--runtime-filename",
        default=DEFAULT_RUNTIME_FILENAME,
        help="Runtime PNG filename.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Image width in pixels.",
    )
    parser.add_argument(
        "--no-global-runtime",
        action="store_true",
        help="Exclude global stages from the accumulated runtime chart.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the standalone performance plotting command."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    result = generate_performance_plots(
        args.source,
        output_directory=args.output_directory,
        timeline_filename=args.timeline_filename,
        runtime_filename=args.runtime_filename,
        include_global_runtime=not args.no_global_runtime,
        image_width=max(900, int(args.width)),
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
    return 0


__all__ = [
    "PERFORMANCE_SCHEMA_NAME",
    "PERFORMANCE_SCHEMA_VERSION",
    "DEFAULT_TIMELINE_FILENAME",
    "DEFAULT_RUNTIME_FILENAME",
    "StageTiming",
    "TimelineSegment",
    "PerformanceData",
    "PerformancePlotResult",
    "extract_run_summary_timings",
    "parse_log_timings",
    "load_performance_data",
    "reconstruct_pose_timeline",
    "aggregate_stage_runtime",
    "render_execution_timeline",
    "render_stage_runtime",
    "generate_performance_plots",
    "performance_data_to_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
