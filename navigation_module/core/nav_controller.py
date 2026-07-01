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


STEP_ADVANCE_M: float = 9.0
ARRIVAL_M: float = 8.0
LOOP_INTERVAL: float = 0.05
GPS_POLL_S: float = 0.5
MAX_USABLE_HDOP: float = 4.5
DEVIATION_M: float = 50.0
DEVIATION_CONFIRM_COUNT: int = 3


class NavController(threading.Thread):

    def __init__(self, nav_queue: queue.Queue, tts_queue: queue.Queue) -> None:
        super().__init__(daemon=True)
        logger.info("初始化核心控制器…")

        self.nav_queue = nav_queue
        self.tts_queue = tts_queue

        self.reader      = GNSSSerialReader()
        self.parser      = NMEAParser()
        self.kalman      = LinearKalmanFilter()
        self.transformer = CoordTransformer()

        self.map_api = AMapProvider()

        config = load_config()
        self.broadcast_distances: list[int] = sorted(
            config['navigation']['broadcast_distances'], reverse=True
        )

        self._pos_lock: threading.Lock = threading.Lock()
        self._gcj_lon:  float | None   = None
        self._gcj_lat:  float | None   = None

        self._last_predict_ts: float | None = None
        self._last_hdop: float = 1.0

        self.state:       str          = "IDLE"
        self.target_name: str | None   = None
        self.target_lon:  float | None = None
        self.target_lat:  float | None = None

        self._steps:           list[dict] = []
        self._step_idx:        int        = 0
        self._announced:       set[int]   = set()
        self._deviation_count: int        = 0
        
    @property
    def current_pos(self) -> tuple[float, float] | None:
        with self._pos_lock:
            if self._gcj_lon is None:
                return None
            return self._gcj_lon, self._gcj_lat

    def force_set_position(self, gcj_lon: float, gcj_lat: float) -> None:
        with self._pos_lock:
            self._gcj_lon = gcj_lon
            self._gcj_lat = gcj_lat

    def _update_gps(self) -> None:
        for _ in range(10):
            raw = self.reader.read_data()
            if not raw:
                break

            parsed = self.parser.parse(raw)
            if not parsed:
                continue

            if parsed.get('type') == 'GGA':
                self._last_hdop = parsed.get('hdop', 99.9)
                self.kalman.set_hdop(self._last_hdop)
                continue

            if parsed.get('type') == 'RMC' and parsed.get('is_valid'):
                if self._last_hdop > MAX_USABLE_HDOP:
                    logger.debug(f"HDOP={self._last_hdop:.1f} 过大, 丢弃本帧定位")
                    continue

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
        if self.current_pos is not None:
            return True
        logger.warning("等待 GPS 定位…")
        self.tts_queue.put("正在等待 GPS 定位，请稍候。")
        while self.current_pos is None:
            time.sleep(GPS_POLL_S)
            self._update_gps()
            if not self.nav_queue.empty():
                return False
        return True

    @staticmethod
    def _get_step_exit(step: dict) -> tuple[float, float] | None:
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
        if not self._steps or self._step_idx >= len(self._steps):
            return haversine_distance(lon, lat, self.target_lon, self.target_lat)

        step    = self._steps[self._step_idx]
        exit_pt = self._get_step_exit(step)

        if exit_pt is None:
            return haversine_distance(lon, lat, self.target_lon, self.target_lat)

        dist_to_exit = haversine_distance(lon, lat, *exit_pt)
        tail_dist    = sum(
            float(s.get('distance', 0)) for s in self._steps[self._step_idx + 1:]
        )
        return dist_to_exit + tail_dist

    def _try_advance_step(self, lon: float, lat: float) -> None:
        advanced = False
        while self._steps and self._step_idx < len(self._steps):
            exit_pt = self._get_step_exit(self._steps[self._step_idx])
            if exit_pt is None:
                break
            if haversine_distance(lon, lat, *exit_pt) > STEP_ADVANCE_M:
                break
            self._step_idx += 1
            advanced = True

        if not advanced:
            return

        self._deviation_count = 0

        if self._step_idx >= len(self._steps):
            return

        next_step   = self._steps[self._step_idx]
        instruction = next_step.get('instruction', str(next_step))
        remaining   = sum(
            float(s.get('distance', 0)) for s in self._steps[self._step_idx:]
        )
        logger.info(f"步骤推进 → 第 {self._step_idx + 1} / {len(self._steps)} 步")
        self.tts_queue.put(f"前方约 {int(round(remaining))} 米，{instruction}")

    def _check_distance_broadcast(self, remaining: float) -> None:
        for threshold in self.broadcast_distances:
            if threshold not in self._announced and remaining <= threshold:
                self._announced.add(threshold)
                logger.info(f"距离播报: 还有 {threshold} 米")
                self.tts_queue.put(f"距目的地还有 {threshold} 米。")

    def _reset_route(self) -> None:
        self._steps           = []
        self._step_idx        = 0
        self._announced       = set()
        self._deviation_count = 0

    def _is_off_route(self, lon: float, lat: float) -> bool:
        if not self._steps or self._step_idx >= len(self._steps):
            return False
        exit_pt = self._get_step_exit(self._steps[self._step_idx])
        if exit_pt is None:
            return False
        try:
            step_dist = float(self._steps[self._step_idx].get('distance', 0)) or 100.0
        except (TypeError, ValueError):
            step_dist = 100.0
        return haversine_distance(lon, lat, *exit_pt) > step_dist + DEVIATION_M

    def _seed_announced_thresholds(self, total_dist) -> None:
        try:
            total = float(total_dist)
        except (TypeError, ValueError):
            return
        self._announced = {t for t in self.broadcast_distances if t >= total}

    def run(self) -> None:
        logger.info("导航后台线程已启动")

        while True:
            time.sleep(LOOP_INTERVAL)

            now = time.monotonic()
            dt = (now - self._last_predict_ts) if self._last_predict_ts is not None else LOOP_INTERVAL
            self._last_predict_ts = now
            self.kalman.predict(dt)

            self._update_gps()

            if self.state == "IDLE":
                try:
                    new_target = self.nav_queue.get_nowait()
                except queue.Empty:
                    continue

                if new_target == "STOP":
                    self.tts_queue.put("当前没有正在进行的导航。")
                    self._reset_route()
                    continue

                self.target_name = new_target
                self._reset_route()
                self.tts_queue.put(f"收到，正在规划去{self.target_name}的路线，请稍候。")
                self.state = "PLANNING"
                logger.info(f"状态 → PLANNING: 目标 '{self.target_name}'")

            elif self.state == "PLANNING":
                if not self._wait_for_fix():
                    self.state = "IDLE"
                    continue

                lon, lat = self.current_pos

                logger.info(f"向高德请求 '{self.target_name}' 的坐标和步行路线…")
                self.target_lon, self.target_lat = self.map_api.get_coordinate_by_name(
                    self.target_name
                )

                if self.target_lon is None:
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

                self._steps = route.get('steps', [])
                total_dist  = route.get('distance_meters', '未知')

                self._seed_announced_thresholds(total_dist)

                first_instr = (
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

            elif self.state == "NAVIGATING":
                pos = self.current_pos
                if pos is None:
                    continue

                lon, lat = pos

                if self._is_off_route(lon, lat):
                    self._deviation_count += 1
                    if self._deviation_count >= DEVIATION_CONFIRM_COUNT:
                        logger.warning("连续偏离路线，触发重规划")
                        self.tts_queue.put("您似乎偏离了路线，正在重新规划…")
                        self._reset_route()
                        self.state = "PLANNING"
                        continue
                else:
                    self._deviation_count = 0

                self._try_advance_step(lon, lat)

                remaining = self._remaining_distance(lon, lat)

                self._check_distance_broadcast(remaining)

                straight_dist = haversine_distance(
                    lon, lat, self.target_lon, self.target_lat
                )
                arrival_threshold = max(ARRIVAL_M, self._last_hdop * 3.0)
                if straight_dist <= arrival_threshold:
                    logger.info(f"已到达目的地: '{self.target_name}'")
                    self.tts_queue.put(
                        f"您已到达{self.target_name}附近，本次导航结束，祝您顺利！"
                    )
                    self._reset_route()
                    self.state = "IDLE"
                    logger.info("状态 → IDLE")
                    continue

                try:
                    new_cmd = self.nav_queue.get_nowait()
                except queue.Empty:
                    new_cmd = None

                if new_cmd == "STOP":
                    self.tts_queue.put(f"已取消前往{self.target_name}的导航。")
                    self._reset_route()
                    self.state = "IDLE"
                    logger.info("用户中止导航 → IDLE")

                elif new_cmd:
                    self.tts_queue.put(
                        f"已取消前往{self.target_name}，重新规划路线中。"
                    )
                    self.target_name = new_cmd
                    self._reset_route()
                    self.state = "PLANNING"
                    logger.info(f"目的地变更 → PLANNING: '{new_cmd}'")


    def start_navigation(self, destination: str) -> dict:
        self.nav_queue.put(destination)
        return {"success": True, "status": f"正在规划前往「{destination}」的路线。"}

    def stop_navigation(self) -> dict:
        self.nav_queue.put("STOP")
        return {"success": True, "status": "导航已取消。"}

    def get_nav_status(self) -> dict:
        pos = self.current_pos

        steps    = self._steps
        step_idx = self._step_idx

        step_info = None
        if steps and step_idx < len(steps):
            step_info = {
                "current": step_idx + 1,
                "total":   len(steps),
                "instruction": steps[step_idx].get('instruction', ''),
            }
        return {
            "state":      self.state,
            "target":     self.target_name,
            "position":   {"lon": pos[0], "lat": pos[1]} if pos else None,
            "step":       step_info,
            "gps_fix":    pos is not None,
        }


    def shutdown(self) -> None:
        if self.reader:
            self.reader.close()
        logger.info("控制器已安全释放资源")