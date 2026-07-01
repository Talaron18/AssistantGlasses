
from __future__ import annotations

import csv
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nav_controller import NavController
from algo.geo.haversine import haversine_distance

HTTP_PORT       = 8800
TRACK_MAXLEN    = 1500
SAMPLE_INTERVAL = 0.5
TRACK_INTERVAL  = 1.0
CLEAR_LINE      = "\x1b[2K\r"

_lock = threading.Lock()
_print_lock = threading.Lock()
_state = {
    "raw_track":      deque(maxlen=TRACK_MAXLEN),
    "filtered_track": deque(maxlen=TRACK_MAXLEN),
    "raw_pos":        None,
    "rmc_times":      deque(maxlen=200),
    "last_rmc_ts":    None,
    "tts_log":        deque(maxlen=10),
}

controller: NavController | None = None


def log_line(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        sys.stdout.write(CLEAR_LINE)
        sys.stdout.write(f"[{ts}] {msg}\n")
        sys.stdout.flush()


def install_raw_fix_recorder(ctrl: NavController) -> None:
    orig_parse = ctrl.parser.parse

    def recording_parse(raw_line):
        parsed = orig_parse(raw_line)
        if parsed and parsed.get("type") == "RMC" and parsed.get("is_valid"):
            glon, glat = ctrl.transformer.wgs84_to_gcj02(
                parsed["longitude"], parsed["latitude"]
            )
            now = time.monotonic()
            with _lock:
                _state["raw_pos"] = (glon, glat)
                _state["raw_track"].append((round(glon, 6), round(glat, 6)))
                _state["rmc_times"].append(now)
                _state["last_rmc_ts"] = now
        return parsed

    ctrl.parser.parse = recording_parse


def tts_consumer(tts_q: queue.Queue) -> None:
    while True:
        msg = tts_q.get()
        log_line(f"🔊 播报 ▶ {msg}")
        ts = datetime.now().strftime("%H:%M:%S")
        with _lock:
            _state["tts_log"].appendleft(f"[{ts}] {msg}")


def collector(ctrl: NavController, stop_evt: threading.Event,
              track_path: str) -> None:
    last_pos = None
    last_track_ts = 0.0

    with open(track_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "lon_gcj02", "lat_gcj02", "state",
                         "step", "remaining_m", "straight_m", "drift_m"])

        while not stop_evt.is_set():
            nav  = ctrl.get_nav_status()
            pos  = nav["position"]
            step = nav["step"]
            state = nav["state"]

            if pos:
                p = (pos["lon"], pos["lat"])
                if p != last_pos:
                    last_pos = p
                    with _lock:
                        _state["filtered_track"].append(
                            (round(p[0], 6), round(p[1], 6))
                        )

            remaining = straight = drift = None
            if pos and state == "NAVIGATING" and ctrl.target_lon is not None:
                remaining = ctrl._remaining_distance(pos["lon"], pos["lat"])
                straight  = haversine_distance(
                    pos["lon"], pos["lat"], ctrl.target_lon, ctrl.target_lat)
            with _lock:
                raw_pos = _state["raw_pos"]
            if pos and raw_pos:
                drift = haversine_distance(
                    raw_pos[0], raw_pos[1], pos["lon"], pos["lat"])

            pos_str  = f"{pos['lon']:.6f},{pos['lat']:.6f}" if pos else "无定位"
            step_str = (f"{step['current']}/{step['total']} "
                        f"{step['instruction'][:12]}") if step else "—"
            rem_str  = f"{remaining:6.0f}m" if remaining is not None else "    —"
            str_str  = f"{straight:6.0f}m"  if straight  is not None else "    —"
            gps_str  = "🟢" if nav["gps_fix"] else "🔴"
            line = (f"  {gps_str} │ {state:<10} │ {pos_str:<21} "
                    f"│ 步骤 {step_str:<18} │ 剩余 {rem_str} │ 直线 {str_str}")
            with _print_lock:
                sys.stdout.write(CLEAR_LINE + line)
                sys.stdout.flush()

            now = time.monotonic()
            if pos and now - last_track_ts >= TRACK_INTERVAL:
                last_track_ts = now
                writer.writerow([
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    f"{pos['lon']:.6f}", f"{pos['lat']:.6f}",
                    state,
                    step["current"] if step else "",
                    f"{remaining:.1f}" if remaining is not None else "",
                    f"{straight:.1f}"  if straight  is not None else "",
                    f"{drift:.1f}"     if drift     is not None else "",
                ])
                f.flush()

            time.sleep(SAMPLE_INTERVAL)


def _planned_route(ctrl: NavController) -> list:
    pts = []
    for step in ctrl._steps:
        polyline = step.get("polyline", "")
        if not polyline:
            continue
        for pair in polyline.strip().rstrip(";").split(";"):
            xy = pair.split(",")
            if len(xy) == 2:
                try:
                    pts.append((round(float(xy[0]), 6), round(float(xy[1]), 6)))
                except ValueError:
                    pass
    return pts


def build_status() -> dict:
    ctrl = controller
    nav = ctrl.get_nav_status()
    pos = nav["position"]
    now = time.monotonic()

    with _lock:
        raw_pos    = _state["raw_pos"]
        raw_track  = list(_state["raw_track"])
        filt_track = list(_state["filtered_track"])
        tts_log    = list(_state["tts_log"])
        last_rmc   = _state["last_rmc_ts"]
        recent = [t for t in _state["rmc_times"] if now - t <= 5.0]
        rmc_hz = round(len(recent) / 5.0, 1)

    drift_m = None
    if pos and raw_pos:
        drift_m = round(haversine_distance(
            raw_pos[0], raw_pos[1], pos["lon"], pos["lat"]), 1)

    remaining_m = straight_m = None
    if pos and nav["state"] == "NAVIGATING" and ctrl.target_lon is not None:
        remaining_m = round(ctrl._remaining_distance(pos["lon"], pos["lat"]), 1)
        straight_m  = round(haversine_distance(
            pos["lon"], pos["lat"], ctrl.target_lon, ctrl.target_lat), 1)

    reader = ctrl.reader
    in_waiting = None
    if reader.is_connected and reader.serial_conn is not None:
        try:
            in_waiting = reader.serial_conn.in_waiting
        except Exception:
            in_waiting = None

    return {
        "state": nav["state"],
        "target": nav["target"],
        "step": nav["step"],
        "filtered_pos": pos,
        "raw_pos": {"lon": raw_pos[0], "lat": raw_pos[1]} if raw_pos else None,
        "target_pos": (
            {"lon": ctrl.target_lon, "lat": ctrl.target_lat}
            if ctrl.target_lon is not None else None
        ),
        "drift_m": drift_m,
        "remaining_m": remaining_m,
        "straight_m": straight_m,
        "hdop": getattr(ctrl, "_last_hdop", None),
        "serial": {
            "connected": reader.is_connected,
            "in_waiting": in_waiting,
            "buffer_len": len(getattr(reader, "_buffer", "")),
            "rmc_hz": rmc_hz,
            "fix_age_s": round(now - last_rmc, 1) if last_rmc else None,
        },
        "raw_track": raw_track,
        "filtered_track": filt_track,
        "route": _planned_route(ctrl),
        "tts_log": tts_log,
    }


PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>导航实地监控</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 body{margin:0;font-family:system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
 #map{flex:1}
 #panel{background:#111;color:#eee;font-size:13px;padding:8px 12px;line-height:1.7;
        max-height:42vh;overflow-y:auto}
 .row{display:flex;flex-wrap:wrap;gap:4px 18px}
 .k{color:#888} .v{font-weight:600}
 .ok{color:#4caf50}.warn{color:#ffb300}.bad{color:#f44336}
 #tts{margin-top:6px;border-top:1px solid #333;padding-top:4px;color:#9cf;font-size:12px}
 #follow{position:absolute;top:10px;right:10px;z-index:1000;background:#fff;
         border:1px solid #aaa;border-radius:4px;padding:4px 10px;cursor:pointer}
</style></head><body>
<div id="map"></div>
<button id="follow">跟随: 开</button>
<div id="legend" style="position:absolute;top:48px;right:10px;z-index:1000;background:#fff;
     border:1px solid #aaa;border-radius:4px;padding:4px 10px;font-size:12px;line-height:1.8">
  <label><input type="checkbox" id="routeChk" checked>
    <span style="color:#2e7d32">━━</span> 规划路线</label><br>
  <label><input type="checkbox" id="rawChk" checked>
    <span style="color:#f44336">━ ╌</span> 原始 GPS</label><br>
  <label><input type="checkbox" id="filtChk" checked>
    <span style="color:#2196f3">━━</span> 滤波输出</label>
</div>
<div id="panel">加载中…</div>
<script>
const map = L.map('map').setView([39.9, 116.4], 17);
// 高德路网瓦片 (GCJ-02), 与系统输出坐标系一致, 直接叠加即可
L.tileLayer('https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
  {subdomains:['1','2','3','4'], maxZoom:19, attribution:'AMap'}).addTo(map);

// 叠放顺序 (底→顶): 绿色规划路线 → 蓝色滤波轨迹 → 红色原始虚线。
// 滤波贴合原始时两线几乎重合, 若红色在下会被蓝色完全盖住而"消失"。
const routeLine = L.polyline([], {color:'#2e7d32', weight:6, opacity:0.45}).addTo(map);
const filtLine = L.polyline([], {color:'#2196f3', weight:4, opacity:0.85}).addTo(map);
const rawLine  = L.polyline([], {color:'#f44336', weight:2, dashArray:'6 6'}).addTo(map);
const filtDot  = L.circleMarker([0,0],{radius:7,color:'#2196f3',fillColor:'#2196f3',fillOpacity:0.9}).addTo(map);
const rawDot   = L.circleMarker([0,0],{radius:4,color:'#f44336',fillColor:'#fff',fillOpacity:1,weight:2}).addTo(map);
let targetMark = null, centered = false, follow = true;

document.getElementById('follow').onclick = function(){
  follow = !follow; this.textContent = '跟随: ' + (follow ? '开' : '关');
};
document.getElementById('routeChk').onchange = function(){
  if(this.checked){ routeLine.addTo(map); routeLine.bringToBack(); }
  else map.removeLayer(routeLine);
};
document.getElementById('rawChk').onchange = function(){
  if(this.checked){ rawLine.addTo(map); rawDot.addTo(map); }
  else { map.removeLayer(rawLine); map.removeLayer(rawDot); }
};
document.getElementById('filtChk').onchange = function(){
  if(this.checked){ filtLine.addTo(map); filtDot.addTo(map); rawLine.bringToFront(); rawDot.bringToFront(); }
  else { map.removeLayer(filtLine); map.removeLayer(filtDot); }
};

function cls(v, warn, bad){ return v==null ? '' : v>=bad ? 'bad' : v>=warn ? 'warn' : 'ok'; }

async function tick(){
  try{
    const s = await (await fetch('/status')).json();
    const ll = p => [p.lat, p.lon];

    if (s.route.length){ routeLine.setLatLngs(s.route.map(p=>[p[1],p[0]])); routeLine.bringToBack(); }
    else routeLine.setLatLngs([]);
    if (s.raw_track.length)      rawLine.setLatLngs(s.raw_track.map(p=>[p[1],p[0]]));
    if (s.filtered_track.length) filtLine.setLatLngs(s.filtered_track.map(p=>[p[1],p[0]]));
    if (s.raw_pos)      rawDot.setLatLng(ll(s.raw_pos));
    if (s.filtered_pos) filtDot.setLatLng(ll(s.filtered_pos));
    if (s.target_pos && !targetMark) targetMark = L.marker(ll(s.target_pos)).addTo(map);
    if (s.target_pos && targetMark)  targetMark.setLatLng(ll(s.target_pos));

    if (s.filtered_pos){
      if (!centered){ map.setView(ll(s.filtered_pos), 18); centered = true; }
      else if (follow) map.panTo(ll(s.filtered_pos), {animate:false});
    }

    const se = s.serial, stp = s.step;
    document.getElementById('panel').innerHTML =
     '<div class="row">'
     +`<span><span class="k">状态</span> <span class="v">${s.state}</span></span>`
     +`<span><span class="k">目标</span> <span class="v">${s.target??'—'}</span></span>`
     +`<span><span class="k">步骤</span> <span class="v">${stp?stp.current+'/'+stp.total+' '+stp.instruction:'—'}</span></span>`
     +`<span><span class="k">剩余</span> <span class="v">${s.remaining_m??'—'} m</span></span>`
     +`<span><span class="k">直线</span> <span class="v">${s.straight_m??'—'} m</span></span>`
     +'</div><div class="row">'
     +`<span><span class="k">漂移(原始↔滤波)</span> <span class="v ${cls(s.drift_m,8,20)}">${s.drift_m??'—'} m</span></span>`
     +`<span><span class="k">HDOP</span> <span class="v ${cls(s.hdop,2.5,5)}">${s.hdop??'—'}</span></span>`
     +`<span><span class="k">原始点</span> <span class="v ${s.raw_track.length?'ok':'bad'}">${s.raw_track.length}</span></span>`
     +`<span><span class="k">滤波点</span> <span class="v">${s.filtered_track.length}</span></span>`
     +'</div><div class="row">'
     +`<span><span class="k">串口</span> <span class="v ${se.connected?'ok':'bad'}">${se.connected?'已连接':'断开'}</span></span>`
     +`<span><span class="k">in_waiting</span> <span class="v ${cls(se.in_waiting,512,2048)}">${se.in_waiting??'—'} B</span></span>`
     +`<span><span class="k">内部缓冲</span> <span class="v ${cls(se.buffer_len,256,1024)}">${se.buffer_len} 字符</span></span>`
     +`<span><span class="k">RMC帧率</span> <span class="v">${se.rmc_hz} Hz</span></span>`
     +`<span><span class="k">定位时延</span> <span class="v ${cls(se.fix_age_s,3,10)}">${se.fix_age_s??'—'} s</span></span>`
     +'</div>'
     +`<div id="tts">${s.tts_log.map(t=>'🔊 '+t).join('<br>')||'(暂无播报)'}</div>`;
  }catch(e){
    document.getElementById('panel').innerHTML =
      '<span class="bad">连接监控服务失败: '+e+'</span>';
  }
}
setInterval(tick, 600); tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.path.startswith("/status"):
            body = json.dumps(build_status(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        else:
            self.send_response(404)
            body = b"not found"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    global controller

    print("=" * 70)
    print(" 盲杖导航 · 实地测试 (终端仪表盘 + 网页地图)")
    print(f" 网页监控:  http://127.0.0.1:{HTTP_PORT}  (手机用电脑局域网 IP)")
    print(" 终端输入目的地回车开始; 行进中输入新地名改道; stop 取消; q 退出")
    print("=" * 70)

    tts_q: queue.Queue = queue.Queue()
    nav_q: queue.Queue = queue.Queue()

    controller = NavController(nav_queue=nav_q, tts_queue=tts_q)
    install_raw_fix_recorder(controller)
    controller.start()

    threading.Thread(target=tts_consumer, args=(tts_q,), daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    track_path = datetime.now().strftime("track_%Y%m%d_%H%M%S.csv")
    log_line(f"轨迹将记录到: {track_path}")
    stop_evt = threading.Event()
    threading.Thread(
        target=collector, args=(controller, stop_evt, track_path), daemon=True
    ).start()

    log_line("等待 GNSS 首次定位 (请到开阔处)…")
    t0 = time.monotonic()
    while controller.current_pos is None:
        if time.monotonic() - t0 > 60:
            log_line("⚠ 60 秒内未获得定位, 仍可继续, 但请检查串口/天线")
            break
        time.sleep(0.5)
    if controller.current_pos:
        lon, lat = controller.current_pos
        log_line(f"✓ 已定位: {lon:.6f}, {lat:.6f}")

    log_line("请输入目的地名称后回车:")
    try:
        while True:
            cmd = input().strip()
            if not cmd:
                continue
            if cmd.lower() in ("q", "quit", "exit"):
                break
            if cmd.lower() == "stop":
                controller.stop_navigation()
                log_line("已发送取消指令")
            else:
                controller.start_navigation(cmd)
                log_line(f"已发送目的地: {cmd}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop_evt.set()
        server.shutdown()
        time.sleep(SAMPLE_INTERVAL + 0.1)
        controller.shutdown()
        print(f"\n测试结束, 轨迹已保存: {track_path}")


if __name__ == "__main__":
    main()