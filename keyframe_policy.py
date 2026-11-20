"""Sparse keyframes for newly generated scenes; never run while loading/saving user files."""
import copy
import functools
import math


def _numbers(value, path=()):
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)) and math.isfinite(value):
        return {path: float(value)}
    if isinstance(value, list):
        result = {}
        for i, item in enumerate(value):
            result.update(_numbers(item, path + (i,)))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key != "t":
                result.update(_numbers(item, path + (key,)))
        return result
    return {}


def within_density(times):
    ordered = sorted(times)
    # Inclusive 10-second windows may contain at most five points.
    return all(ordered[i + 5] - ordered[i] > 10.0 + 1e-6
               for i in range(len(ordered) - 5))


def sparse_track(raw, duration):
    if not raw:
        return []
    by_time = {float(k.get("t", 0)): copy.deepcopy(k) for k in raw}
    keys = [by_time[t] for t in sorted(by_time)]
    budget = max(2, int(math.ceil(max(0.01, float(duration)) / 10.0 * 5)))
    if len(keys) <= budget and within_density([k["t"] for k in keys]):
        return keys
    retained = {0, len(keys) - 1}
    vectors = [_numbers(k) for k in keys]
    while len(retained) < min(budget, len(keys)):
        best = None
        best_score = -1.0
        ordered = sorted(retained)
        for left, right in zip(ordered, ordered[1:]):
            dt = float(keys[right]["t"]) - float(keys[left]["t"])
            for i in range(left + 1, right):
                times = [keys[j]["t"] for j in retained] + [keys[i]["t"]]
                if not within_density(times):
                    continue
                u = (float(keys[i]["t"]) - float(keys[left]["t"])) / max(dt, 1e-9)
                error = 0.0
                for channel, val in vectors[i].items():
                    if channel not in vectors[left] or channel not in vectors[right]:
                        continue
                    start, end = vectors[left][channel], vectors[right][channel]
                    if any("rot" in str(part) for part in channel):
                        delta = (end - start + 180.0) % 360.0 - 180.0
                        residual = (val - (start + u * delta) + 180.0) % 360.0 - 180.0
                        residual /= 30.0
                    else:
                        residual = val - (start + u * (end - start))
                        if "fov" in channel or "focal_mm" in channel:
                            residual /= 30.0
                    error += residual * residual
                transition = any(keys[i].get(k) != keys[i - 1].get(k)
                                 for k in ("visible", "hidden", "has_pos", "has_fov", "pose_name"))
                priority = bool(keys[i].get("stop") or keys[i].get("pause") or transition)
                score = (1000000.0 if priority else 0.0) + error
                if score > best_score:
                    best, best_score = i, score
        if best is None or best_score < 1e-10:
            break
        retained.add(best)
    return [keys[i] for i in sorted(retained)]


def sparsify_scene(scene):
    result = copy.deepcopy(scene)
    duration = float(result.get("duration", 10) or 10)
    before = len(result.get("camera_keyframes") or [])
    result["camera_keyframes"] = sparse_track(result.get("camera_keyframes") or [], duration)
    after = len(result["camera_keyframes"])
    for obj in result.get("objects", []):
        if obj.get("keyframes"):
            before += len(obj["keyframes"])
            obj["keyframes"] = sparse_track(obj["keyframes"], duration)
            after += len(obj["keyframes"])
    return result, {"before": before, "after": after, "max_per_10_seconds": 5}


def sparse_generated_scene(builder):
    @functools.wraps(builder)
    def wrapped(*args, **kwargs):
        return sparsify_scene(builder(*args, **kwargs))[0]
    return wrapped
