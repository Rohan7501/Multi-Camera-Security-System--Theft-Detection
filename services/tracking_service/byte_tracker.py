"""ByteTrack multi-object tracker, self-contained.

Faithful port of Zhang et al. (2022): a constant-velocity Kalman filter on
(cx, cy, aspect, height), then two-stage IoU association -- match the confident
detections first, then use the low-confidence boxes everyone else throws away to
recover tracks that would otherwise be lost. One instance per stream.

Needs numpy and scipy. Tuning knobs and gotchas are in README.md.
"""
import numpy as np

from tracker_base import Tracker, TrackerInput

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _HAVE_SCIPY = True
except Exception:                        # pragma: no cover - fallback path
    _HAVE_SCIPY = False


# ---- Kalman filter (SORT/ByteTrack 8-state: xyah + velocities) --------------

class KalmanFilter:
    """Tracks box state [cx, cy, a, h, vx, vy, va, vh] with a constant-velocity model."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        # Uncertainty scales relative to box height (bigger boxes -> looser gates).
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros(4)]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        # Kalman gain K = P Hᵀ S⁻¹; solve S Kᵀ = H Pᵀ instead of inverting (4x4, np only).
        b = (covariance @ self._update_mat.T).T
        kalman_gain = np.linalg.solve(projected_cov, b).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


# ---- track object -----------------------------------------------------------

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    _shared_kalman = KalmanFilter()
    _count = 0                            # global monotonic id (unique across streams)

    def __init__(self, tlwh, score, cls, det_index=-1):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = float(score)
        self.cls = int(cls)
        self.det_index = int(det_index)   # index of the detection this track holds THIS frame
        self.track_id = 0
        self.state = TrackState.New
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def multi_predict(stracks):
        for st in stracks:
            if st.state != TrackState.Tracked:
                st.mean[7] = 0            # freeze height-velocity while coasting
            st.mean, st.covariance = st.kalman_filter.predict(st.mean, st.covariance)

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True     # first frame: confirm immediately
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.det_index = new_track.det_index

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.det_index = new_track.det_index

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]                 # a * h -> w
        ret[:2] -= ret[2:] / 2           # center -> top-left
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret


# ---- association helpers ----------------------------------------------------

def _ious(atlbrs, btlbrs):
    out = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if out.size == 0:
        return out
    a = np.ascontiguousarray(atlbrs, dtype=np.float32)
    b = np.ascontiguousarray(btlbrs, dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def iou_distance(atracks, btracks):
    atlbrs = [t.tlbr for t in atracks]
    btlbrs = [t.tlbr for t in btracks]
    return 1.0 - _ious(atlbrs, btlbrs)   # cost = 1 - IoU


def _greedy_assignment(cost, thresh):
    matches = []
    used_r, used_c = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in order:
        if cost[r, c] > thresh:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        matches.append([r, c])
    return matches


def linear_assignment(cost_matrix, thresh):
    """Return (matches Kx2, unmatched_rows, unmatched_cols) under an IoU-cost gate."""
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                list(range(cost_matrix.shape[0])),
                list(range(cost_matrix.shape[1])))

    if _HAVE_SCIPY:
        # Push above-threshold pairs out of contention, then filter the result.
        cost = cost_matrix.copy()
        cost[cost > thresh] = thresh + 1e-4
        row_ind, col_ind = _lsa(cost)
        matches = [[r, c] for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= thresh]
    else:                                # pragma: no cover
        matches = _greedy_assignment(cost_matrix, thresh)

    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    unmatched_a = [r for r in range(cost_matrix.shape[0]) if r not in mr]
    unmatched_b = [c for c in range(cost_matrix.shape[1]) if c not in mc]
    matches = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_a, unmatched_b


def joint_stracks(alist, blist):
    exists, res = {}, []
    for t in alist:
        exists[t.track_id] = True
        res.append(t)
    for t in blist:
        if not exists.get(t.track_id):
            exists[t.track_id] = True
            res.append(t)
    return res


def sub_stracks(alist, blist):
    d = {t.track_id: t for t in alist}
    for t in blist:
        d.pop(t.track_id, None)
    return list(d.values())


def remove_duplicate_stracks(sa, sb):
    pdist = iou_distance(sa, sb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        timep = sa[p].frame_id - sa[p].start_frame
        timeq = sb[q].frame_id - sb[q].start_frame
        if timep > timeq:
            dupb.add(q)
        else:
            dupa.add(p)
    resa = [t for i, t in enumerate(sa) if i not in dupa]
    resb = [t for i, t in enumerate(sb) if i not in dupb]
    return resa, resb


# ---- the tracker ------------------------------------------------------------

class BYTETracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_thresh = track_thresh              # >= this: high-confidence
        self.det_thresh = track_thresh + 0.1          # spawn new tracks only above this
        self.match_thresh = match_thresh              # IoU gate for stage-1 association
        self.low_thresh = 0.1                         # discard detections below this
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilter()

    def update(self, dets, classes, scores):
        """dets: (N,4) tlbr; classes: (N,); scores: (N,). Returns online STracks."""
        self.frame_id += 1
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(classes).reshape(-1)
        idx = np.arange(len(scores))

        activated, refind, lost, removed = [], [], [], []

        high = scores >= self.track_thresh
        low = (scores > self.low_thresh) & (scores < self.track_thresh)

        def make(sel):
            return [STrack(STrack.tlbr_to_tlwh(t), s, c, i)
                    for t, s, c, i in zip(dets[sel], scores[sel], classes[sel], idx[sel])]

        detections = make(high)
        detections_low = make(low)

        # split current tracks into confirmed vs. tentative (unconfirmed)
        unconfirmed, tracked = [], []
        for t in self.tracked_stracks:
            (tracked if t.is_activated else unconfirmed).append(t)

        # predict all confirmed + lost tracks forward one step
        strack_pool = joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # --- stage 1: high-confidence detections vs. predicted tracks ---
        dists = iou_distance(strack_pool, detections)
        matches, u_track, u_det = linear_assignment(dists, self.match_thresh)
        for it, idet in matches:
            track, det = strack_pool[it], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # --- stage 2: low-confidence detections vs. still-unmatched TRACKED tracks ---
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked, detections_low)
        matches, u_track2, _ = linear_assignment(dists, 0.5)
        for it, idet in matches:
            track, det = r_tracked[it], detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # --- unconfirmed tracks vs. leftover high-confidence detections ---
        detections = [detections[i] for i in u_det]
        dists = iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_det = linear_assignment(dists, 0.7)
        for it, idet in matches:
            unconfirmed[it].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)

        # --- spawn new tracks from remaining strong detections ---
        for inew in u_det:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        # --- age out lost tracks ---
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # --- merge bookkeeping lists ---
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        if len(self.removed_stracks) > 1000:         # bound memory
            self.removed_stracks = self.removed_stracks[-1000:]

        return [t for t in self.tracked_stracks if t.is_activated]


# ---- algorithm-agnostic adapter --------------------------------------------

class ByteTrackTracker(Tracker):
    """ByteTrack behind the algorithm-agnostic Tracker interface (one per stream).

    Ignores inp.frame -- ByteTrack is motion/IoU only, no appearance features.
    """

    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30):
        self._impl = BYTETracker(track_thresh=track_thresh, match_thresh=match_thresh,
                                 track_buffer=track_buffer, frame_rate=frame_rate)

    def update(self, inp: TrackerInput) -> list:
        ids = [-1] * len(inp.boxes)
        online = self._impl.update(inp.boxes, inp.classes, inp.scores)
        for t in online:                             # map each online track to its detection
            if 0 <= t.det_index < len(ids):
                ids[t.det_index] = t.track_id
        return ids
