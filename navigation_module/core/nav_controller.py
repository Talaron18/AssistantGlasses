"""
NavController — blind-cane navigation state machine (background thread).

Fixes and improvements over v1
────────────────────────────────
Thread-safe GPS      – _pos_lock guards every read/write of the current position;
                       PLANNING polls until a valid fix arrives before touching AMap.

Step-by-step nav     – _step_idx tracks the active route step.  When the user
                       walks within STEP_ADVANCE_M of a step's exit waypoint, the
                       next step's instruction is announced automatically.

Trajectory distance  – remaining = haversine(pos → step_exit) + Σ(tail step
                       distances), not haversine(pos → final target).  Falls back
                       to polyline-parsed exit points when AMapProvider does not
                       populate exit_lon/exit_lat directly.

Threshold broadcasts – "passed-below" detection with an _announced set that resets
                       per route; each threshold fires exactly once, with no
                       blocking sleep inside the hot loop.

Queue hygiene        – mid-route interrupt consumes and dispatches the command
                       inline; new destination immediately re-enters PLANNING with
                       a user-facing announcement.

Public tool API      – start_navigation / stop_navigation / get_nav_status can be
                       called directly by Gemma4Agent's tool dispatcher.
"""

from __future__ import annotations

import time
import sys
import os
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.gnss.nmea_parser import NMEAParser
from algo.fusion.linear_kalman import LinearKalmanFilter
from algo.geo.coord_transform import CoordTransformer
from algo.geo.haversine import haversine_distance
from services.amap_provider import AMapProvider
from config.config_loader import load_config
from sensors.gnss.serial_reader import GNSSSerialReader
from utils.logger import get_logger

logger = get_logger(__name__)

# ── tunable constants ──────────────────────────────────────────────────────────
# Pedestrian walks ~1.4 m/s; 15 m gives ~10 s of advance warning for a turn.
STEP_ADVANCE_M: float = 15.0
# Straight-line distance to the final target at which we declare arrival.
ARRIVAL_M: float = 8.0
# Loop interval (seconds). 50 ms → 20 Hz, plenty for walking navigation.
LOOP_INTERVAL: float = 0.05
# How long to wait between GPS-fix polls while in PLANNING.
GPS_POLL_S: float = 0.5


class NavController(threading.Thread):
    """
    盲人智能拐杖导航系统 (多线程状态机)

    State machine
    ─────────────
    IDLE → PLANNING → NAVIGATING → IDLE
            ↑___________________________↑  (arrival / stop / new destination)

    Thread-safety note
    ──────────────────
    GPS position (_gcj_lon / _gcj_lat) is written by the run() loop and may be
    read from outside (e.g. get_nav_status).  All accesses go through _pos_lock.

    Integration with Gemma4Agent
    ─────────────────────────────
    Pass the NavController instance to Gemma4Agent(nav_controller=...).
    The agent automatically registers start_navigation / stop_navigation /
    get_nav_status as LLM-callable tools and delegates calls to the methods
    defined at the bottom of this class.
    """

    def __init__(self, nav_queue: queue.Queue, tts_queue: queue.Queue) -> None:
        super().__init__(daemon=True)
        logger.info("初始化核心控制器…")

        # ── inter-thread communication ─────────────────────────────────────
        self.nav_queue = nav_queue    # receives destination strings or "STOP"
        self.tts_queue = tts_queue    # sends speech strings out

        # ── sensors & algorithms ───────────────────────────────────────────
        self.reader      = GNSSSerialReader()
        self.parser      = NMEAParser()
        self.kalman      = LinearKalmanFilter()
        self.transformer = CoordTransformer()

        # ── cloud service ──────────────────────────────────────────────────
        self.map_api = AMapProvider()

        # ── config ────────────────────────────────────────────────────────
        config = load_config()
        # Store descending so we can pop announced thresholds in order.
        self.broadcast_distances: list[int] = sorted(
            config['navigation']['broadcast_distances'], reverse=True
        )

        # ── thread-safe GPS position ───────────────────────────────────────
        self._pos_lock: threading.Lock = threading.Lock()
        self._gcj_lon:  float | None   = None
        self._gcj_lat:  float | None   = None

        # ── state machine ──────────────────────────────────────────────────
        self.state:       str        = "IDLE"
        self.target_name: str | None = None
        self.target_lon:  float | None = None
        self.target_lat:  float | None = None

        # ── route tracking (reset on each new route) ───────────────────────
        self._steps:     list[dict] = []   # step dicts from AMap
        self._step_idx:  int        = 0    # index of the step being navigated
        self._announced: set[int]   = set()  # thresholds already spoken

    # ──────────────────────────────────────────────────────────────────────────
    # GPS helpers
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def current_pos(self) -> tuple[float, float] | None:
        """Return (gcj_lon, gcj_lat) or None if no fix yet."""
        with self._pos_lock:
            if self._gcj_lon is None:
                return None
            return self._gcj_lon, self._gcj_lat
        
    def force_set_position(self, gcj_lon: float, gcj_lat: float) -> None:
        """【地下室专用测试】强制改写当前位置，模拟人已经走到这里"""
        with self._pos_lock:
            # 【新增判断】如果是第一次获取到位置，才初始化卡尔曼滤波器
            if self._gcj_lon is None:
                self.kalman.initialize(gcj_lon, gcj_lat)
                
            self._gcj_lon = gcj_lon
            self._gcj_lat = gcj_lat

    def _update_gps(self) -> None:
        """读取 NMEA 数据，并排干缓冲区防止历史数据积压导致定位滞后"""
        for _ in range(50):
            raw = self.reader.read_data()
            if not raw:
                break  # 缓冲区已经读空，跳出循环
                
            parsed = self.parser.parse(raw)
            if (
                parsed
                and parsed.get('type') == 'RMC'
                and parsed.get('is_valid')
            ):
                self.kalman.update((
                    parsed['longitude'],
                    parsed['latitude'],
                    parsed['speed_kmh'],
                    parsed['true_course'],
                ))
                lon, lat = self.kalman.get_state()
                gcj_lon, gcj_lat = self.transformer.wgs84_to_gcj02(lon, lat)
                with self._pos_lock:
                    self._gcj_lon = gcj_lon
                    self._gcj_lat = gcj_lat

    def _wait_for_fix(self) -> bool:
        """
        Block (with GPS polling) until a valid position is available.
        Returns True when a fix is acquired, False if a queue command interrupts.
        """
        if self.current_pos is not None:
            return True
        logger.warning("等待 GPS 定位…")
        self.tts_queue.put("正在等待 GPS 定位，请稍候。")
        while self.current_pos is None:
            time.sleep(GPS_POLL_S)
            self._update_gps()
            if not self.nav_queue.empty():
                return False   # user sent a new command; bail out
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Route step helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_step_exit(step: dict) -> tuple[float, float] | None:
        """
        Return the (lon, lat) exit waypoint for a route step.

        Tries two sources in order:
        1. Explicit exit_lon / exit_lat fields (set by AMapProvider).
        2. Last coordinate pair in the 'polyline' string
           (AMap format: "lon1,lat1;lon2,lat2;…").
        """
        if step.get('exit_lon') and step.get('exit_lat'):
            return step['exit_lon'], step['exit_lat']

        polyline: str = step.get('polyline', '')
        if polyline:
            points = polyline.strip().rstrip(';').split(';')
            if points:
                pair = points[-1].split(',')
                if len(pair) == 2:
                    try:
                        return float(pair[0]), float(pair[1])
                    except ValueError:
                        pass
        return None

    def _remaining_distance(self, lon: float, lat: float) -> float:
        """
        Trajectory-based remaining distance (metres).

        = haversine(current_pos → current step exit waypoint)
          + sum of all subsequent step distances.

        Falls back to haversine(current_pos → target) when no waypoint data
        is available (e.g. AMap returned no polyline for any step).
        """
        if not self._steps or self._step_idx >= len(self._steps):
            # All steps done or no step data — use straight-line as best guess
            return haversine_distance(lon, lat, self.target_lon, self.target_lat)

        step     = self._steps[self._step_idx]
        exit_pt  = self._get_step_exit(step)

        if exit_pt is None:
            # No waypoint — degrade to haversine of target
            return haversine_distance(lon, lat, self.target_lon, self.target_lat)

        dist_to_exit = haversine_distance(lon, lat, *exit_pt)
        tail_dist    = sum(
            float(s.get('distance', 0)) for s in self._steps[self._step_idx + 1:]
        )
        return dist_to_exit + tail_dist

    def _try_advance_step(self, lon: float, lat: float) -> None:
        """
        If the user is within STEP_ADVANCE_M of the current step's exit
        waypoint, advance to the next step and speak its instruction.
        Advances at most one step per call to stay in sync with the walk.
        """
        if not self._steps or self._step_idx >= len(self._steps):
            return

        exit_pt = self._get_step_exit(self._steps[self._step_idx])
        if exit_pt is None:
            return

        if haversine_distance(lon, lat, *exit_pt) > STEP_ADVANCE_M:
            return

        self._step_idx += 1
        if self._step_idx >= len(self._steps):
            return   # last step done; arrival check handles the rest

        next_step   = self._steps[self._step_idx]
        instruction = next_step.get('instruction', str(next_step))
        remaining   = sum(
            float(s.get('distance', 0)) for s in self._steps[self._step_idx:]
        )
        logger.info(f"步骤推进 → 第 {self._step_idx + 1} / {len(self._steps)} 步")
        self.tts_queue.put(f"前方约 {remaining} 米，{instruction}")

    def _check_distance_broadcast(self, remaining: float) -> None:
        """
        Announce each distance threshold exactly once per route,
        the first time `remaining` passes below it.
        """
        for threshold in self.broadcast_distances:
            if threshold not in self._announced and remaining <= threshold:
                self._announced.add(threshold)
                logger.info(f"距离播报: 还有 {threshold} 米")
                self.tts_queue.put(f"距目的地还有 {threshold} 米。")

    def _reset_route(self) -> None:
        """Clear all per-route state. Call before starting every new route."""
        self._steps     = []
        self._step_idx  = 0
        self._announced = set()

    # ──────────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("导航后台线程已启动")

        while True:
            time.sleep(LOOP_INTERVAL)
            self._update_gps()

            # ── IDLE ──────────────────────────────────────────────────────────
            if self.state == "IDLE":
                try:
                    new_target = self.nav_queue.get_nowait()
                except queue.Empty:
                    continue   # nothing to do

                if new_target == "STOP":
                    self.tts_queue.put("当前没有正在进行的导航。")
                    self._reset_route()
                    continue

                # New destination received
                self.target_name = new_target
                self._reset_route()
                self.tts_queue.put(f"收到，正在规划去{self.target_name}的路线，请稍候。")
                self.state = "PLANNING"
                logger.info(f"状态 → PLANNING: 目标 '{self.target_name}'")

            # ── PLANNING ──────────────────────────────────────────────────────
            elif self.state == "PLANNING":
                # Guard: must have a GPS fix before calling the routing API
                if not self._wait_for_fix():
                    # Queue had a new command; drop back to IDLE to process it
                    self.state = "IDLE"
                    continue

                lon, lat = self.current_pos

                logger.info(f"向高德请求 '{self.target_name}' 的坐标和步行路线…")
                self.target_lon, self.target_lat = self.map_api.get_coordinate_by_name(
                    self.target_name
                )

                if not self.target_lon:
                    logger.error(f"找不到地点: '{self.target_name}'")
                    self.tts_queue.put(
                        f"抱歉，在地图上找不到「{self.target_name}」，请重新告诉我目的地。"
                    )
                    self.state = "IDLE"
                    continue

                route = self.map_api.get_walking_route(
                    lon, lat, self.target_lon, self.target_lat
                )
                if not route:
                    logger.error("步行路径规划失败")
                    self.tts_queue.put("路径规划失败，可能是网络问题，请稍后重试。")
                    self.state = "IDLE"
                    continue

                # ── Route acquired ─────────────────────────────────────────
                self._steps    = route.get('steps', [])
                total_dist     = route.get('distance_meters', '未知')
                first_instr    = (
                    self._steps[0].get('instruction', str(self._steps[0]))
                    if self._steps else "沿当前方向前行"
                )
                self.tts_queue.put(
                    f"路线规划成功，步行总距离约 {total_dist} 米。"
                    f"第一步：{first_instr}"
                )
                self.state = "NAVIGATING"
                logger.info(
                    f"状态 → NAVIGATING，共 {len(self._steps)} 步，"
                    f"总距离 {total_dist} 米"
                )

            # ── NAVIGATING ────────────────────────────────────────────────────
            elif self.state == "NAVIGATING":
                pos = self.current_pos
                if pos is None:
                    continue   # waiting for GPS fix, stay quiet

                lon, lat = pos

                # ① Advance to next step when close enough to current exit point
                self._try_advance_step(lon, lat)

                # ② Compute trajectory-based remaining distance
                remaining = self._remaining_distance(lon, lat)

                # ③ Distance threshold announcements (each threshold fires once)
                self._check_distance_broadcast(remaining)

                # ④ Arrival detection (straight-line, last few metres)
                straight_dist = haversine_distance(
                    lon, lat, self.target_lon, self.target_lat
                )
                if straight_dist <= ARRIVAL_M:
                    logger.info(f"已到达目的地: '{self.target_name}'")
                    self.tts_queue.put(
                        f"您已到达{self.target_name}附近，本次导航结束，祝您顺利！"
                    )
                    self._reset_route()
                    self.state = "IDLE"
                    logger.info("状态 → IDLE")
                    continue

                # ⑤ Mid-route interrupt: new destination or STOP command
                try:
                    new_cmd = self.nav_queue.get_nowait()
                except queue.Empty:
                    new_cmd = None

                if new_cmd == "STOP":
                    self.tts_queue.put(f"已取消前往{self.target_name}的导航。")
                    self._reset_route()
                    self.state = "IDLE"
                    logger.info("用户中止导航 → IDLE")

                elif new_cmd:   # new destination string
                    self.tts_queue.put(
                        f"已取消前往{self.target_name}，重新规划路线中。"
                    )
                    self.target_name = new_cmd
                    self._reset_route()
                    self.state = "PLANNING"
                    logger.info(f"目的地变更 → PLANNING: '{new_cmd}'")

    # ──────────────────────────────────────────────────────────────────────────
    # Public tool API  (called by Gemma4Agent's tool dispatcher)
    # ──────────────────────────────────────────────────────────────────────────

    def start_navigation(self, destination: str) -> dict:
        """
        Enqueue a navigation request.  Returns immediately; the background
        thread handles planning and route acquisition asynchronously.
        """
        self.nav_queue.put(destination)
        return {"success": True, "status": f"正在规划前往「{destination}」的路线。"}

    def stop_navigation(self) -> dict:
        """Cancel the current navigation."""
        self.nav_queue.put("STOP")
        return {"success": True, "status": "导航已取消。"}

    def get_nav_status(self) -> dict:
        """Return a snapshot of the current navigation state (thread-safe)."""
        pos = self.current_pos
        step_info = None
        if self._steps and self._step_idx < len(self._steps):
            step_info = {
                "current": self._step_idx + 1,
                "total":   len(self._steps),
                "instruction": self._steps[self._step_idx].get('instruction', ''),
            }
        return {
            "state":      self.state,
            "target":     self.target_name,
            "position":   {"lon": pos[0], "lat": pos[1]} if pos else None,
            "step":       step_info,
            "gps_fix":    pos is not None,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release hardware resources gracefully."""
        if self.reader:
            self.reader.close()
        logger.info("控制器已安全释放资源")
