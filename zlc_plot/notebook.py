"""Notebook raster adapter.

The kernel owns a :class:`RasterPlotHost`; the browser receives complete RGBA
fronts and draws transient selector geometry on a second canvas.  Matplotlib
never owns notebook input and no notebook backend is imported by this module.
"""

from __future__ import annotations

from concurrent.futures import Future
import json
import threading
from typing import Any, Mapping

from ._axis_transform import AxisTransform
from ._selector_scene import (
    ColorLimitCandidate,
    SelectorLine,
    SelectorMarkers,
    SelectorScene,
    SelectorSceneKind,
    SelectorTarget,
    SelectorText,
)
from .backends import BackendUnavailableError
from .raster import RasterFront, RasterOperation, RasterPlotHost
from .selectors import SelectorKind


def _target_to_dict(target: SelectorTarget) -> dict[str, object]:
    return {"role": target.role, "cell_index": target.cell_index}


def selector_scene_to_dict(scene: SelectorScene | None) -> dict[str, object]:
    """Encode one selector scene as JSON-compatible primitives."""

    if scene is None:
        return {"groups": []}
    groups: list[dict[str, object]] = []
    for kind, primitives in scene.groups:
        encoded: list[dict[str, object]] = []
        for primitive in primitives:
            if isinstance(primitive, SelectorLine):
                encoded.append({
                    "type": "line",
                    "key": primitive.key,
                    "target": _target_to_dict(primitive.target),
                    "points": [list(point) for point in primitive.points],
                    "color": list(primitive.color),
                    "width_pt": primitive.width_pt,
                    "zorder": primitive.zorder,
                    "linestyle": primitive.linestyle,
                })
            elif isinstance(primitive, SelectorMarkers):
                encoded.append({
                    "type": "markers",
                    "key": primitive.key,
                    "target": _target_to_dict(primitive.target),
                    "points": [list(point) for point in primitive.points],
                    "shape": primitive.shape,
                    "size_pt": primitive.size_pt,
                    "facecolor": list(primitive.facecolor),
                    "edgecolor": None if primitive.edgecolor is None else list(primitive.edgecolor),
                    "edge_width_pt": primitive.edge_width_pt,
                    "zorder": primitive.zorder,
                })
            elif isinstance(primitive, SelectorText):
                encoded.append({
                    "type": "text",
                    "key": primitive.key,
                    "target": _target_to_dict(primitive.target),
                    "text": primitive.text,
                    "position": list(primitive.position),
                    "horizontal_alignment": primitive.horizontal_alignment,
                    "color": list(primitive.color),
                    "font_family": primitive.font_family,
                    "font_size_pt": primitive.font_size_pt,
                    "zorder": primitive.zorder,
                })
            else:
                raise TypeError("unsupported selector scene primitive")
        groups.append({"kind": kind.value, "primitives": encoded})
    return {"groups": groups}


def _target_from_dict(value: Mapping[str, object]) -> SelectorTarget:
    return SelectorTarget(str(value["role"]), value.get("cell_index"))


def selector_scene_from_dict(value: Mapping[str, object]) -> SelectorScene:
    """Decode the JSON-compatible selector scene contract."""

    groups: list[tuple[SelectorKind | SelectorSceneKind, tuple[object, ...]]] = []
    for group in value.get("groups", ()):
        if not isinstance(group, Mapping):
            raise TypeError("selector scene groups must be mappings")
        raw_kind = str(group["kind"])
        try:
            kind: SelectorKind | SelectorSceneKind = SelectorKind(raw_kind)
        except ValueError:
            kind = SelectorSceneKind(raw_kind)
        primitives: list[object] = []
        for raw in group.get("primitives", ()):
            if not isinstance(raw, Mapping):
                raise TypeError("selector scene primitives must be mappings")
            target = _target_from_dict(raw["target"])
            points = tuple(tuple(map(float, point)) for point in raw.get("points", ()))
            primitive_type = str(raw["type"])
            if primitive_type == "line":
                primitives.append(SelectorLine(
                    str(raw["key"]), target, points,
                    tuple(map(float, raw["color"])), float(raw["width_pt"]),
                    float(raw["zorder"]), str(raw.get("linestyle", "-")),
                ))
            elif primitive_type == "markers":
                edge = raw.get("edgecolor")
                primitives.append(SelectorMarkers(
                    str(raw["key"]), target, points, str(raw["shape"]),
                    float(raw["size_pt"]), tuple(map(float, raw["facecolor"])),
                    None if edge is None else tuple(map(float, edge)),
                    float(raw["edge_width_pt"]), float(raw["zorder"]),
                ))
            elif primitive_type == "text":
                primitives.append(SelectorText(
                    str(raw["key"]), target, str(raw["text"]),
                    tuple(map(float, raw["position"])),
                    str(raw["horizontal_alignment"]), tuple(map(float, raw["color"])),
                    str(raw["font_family"]), float(raw["font_size_pt"]),
                    float(raw["zorder"]),
                ))
            else:
                raise ValueError(f"unknown selector scene primitive {primitive_type!r}")
        groups.append((kind, tuple(primitives)))
    return SelectorScene(tuple(groups))


def _axis_to_dict(axis: AxisTransform) -> dict[str, object]:
    return {
        "role": axis.role,
        "cell_index": axis.cell_index,
        "bounds": list(axis.bounds),
        "x_limits": list(axis.x_limits),
        "y_limits": list(axis.y_limits),
        "canonical_x_limits": list(axis.canonical_x_limits),
        "canonical_y_limits": list(axis.canonical_y_limits),
    }


_WIDGET_CLASS: type[Any] | None = None
_WIDGET_MODULE_LOADED = False


def _widget_class() -> type[Any]:
    global _WIDGET_CLASS
    if _WIDGET_CLASS is not None:
        return _WIDGET_CLASS
    try:
        import ipywidgets as widgets
        from traitlets import Bytes, Int, Unicode
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailableError(
            "NotebookView requires ipywidgets; install zlc-plot[notebook]."
        ) from error

    class RasterWidget(widgets.DOMWidget):
        _view_name = Unicode("ZlcRasterView").tag(sync=True)
        _view_module = Unicode("zlc_plot_raster").tag(sync=True)
        _view_module_version = Unicode("1.0.0").tag(sync=True)
        width = Int(1).tag(sync=True)
        height = Int(1).tag(sync=True)
        logical_width = Int(1).tag(sync=True)
        logical_height = Int(1).tag(sync=True)
        logical_dpi = Unicode("96").tag(sync=True)
        device_pixel_ratio = Unicode("1").tag(sync=True)
        rgba = Bytes(b"").tag(sync=True)
        axes_json = Unicode("[]").tag(sync=True)
        scene_json = Unicode('{"groups":[]}').tag(sync=True)

    _WIDGET_CLASS = RasterWidget
    return RasterWidget


_WIDGET_JS = r"""
(function () {
  const defineFn = window.define || (window.requirejs && window.requirejs.define);
  if (!defineFn) return;
  try {
    defineFn('zlc_plot_raster', ['@jupyter-widgets/base'], function (widgets) {
      class ZlcRasterView extends widgets.DOMWidgetView {
        render() {
          this.el.classList.add('zlc-raster-root');
          this.el.style.position = 'relative';
          this.el.style.display = 'inline-block';
          this.canvas = document.createElement('canvas');
          this.overlay = document.createElement('canvas');
          for (const item of [this.canvas, this.overlay]) {
            item.style.position = 'absolute'; item.style.left = '0'; item.style.top = '0';
            item.style.touchAction = 'none'; item.style.userSelect = 'none';
            this.el.appendChild(item);
          }
          this._dragging = false;
          this._button = null;
          this._lastMoveSent = 0;
          this._bind();
          this.listenTo(this.model, 'change:rgba change:axes_json change:scene_json', () => this._paint());
          this._paint();
        }
        _bytes(value) {
          if (value instanceof Uint8Array) return value;
          if (value instanceof ArrayBuffer) return new Uint8Array(value);
          if (typeof value === 'string') {
            const raw = atob(value); const result = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) result[i] = raw.charCodeAt(i);
            return result;
          }
          return new Uint8Array(value || []);
        }
        _paint() {
          const w = this.model.get('width'), h = this.model.get('height');
          const lw = this.model.get('logical_width'), lh = this.model.get('logical_height');
          for (const canvas of [this.canvas, this.overlay]) {
            canvas.width = w; canvas.height = h;
            canvas.style.width = lw + 'px'; canvas.style.height = lh + 'px';
          }
          this.el.style.width = lw + 'px'; this.el.style.height = lh + 'px';
          const bytes = this._bytes(this.model.get('rgba'));
          if (bytes.length === w * h * 4) {
            this.canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(bytes), w, h), 0, 0);
          }
          this._paintScene();
        }
        _axisFor(target) {
          const axes = JSON.parse(this.model.get('axes_json') || '[]');
          return axes.find(a => a.role === target.role && a.cell_index === target.cell_index) ||
                 axes.find(a => a.role === target.role) || null;
        }
        _point(axis, point) {
          const [l,t,r,b] = axis.bounds;
          const [x0,x1] = axis.x_limits, [y0,y1] = axis.y_limits;
          const tx = (point[0] - x0) / (x1 - x0);
          const ty = (y1 - point[1]) / (y1 - y0);
          return [(l + tx * (r-l)) * this.canvas.width,
                  (t + ty * (b-t)) * this.canvas.height];
        }
        _pointAxes(axis, point) {
          const [l,t,r,b] = axis.bounds;
          return [(l + point[0] * (r-l)) * this.canvas.width,
                  (t + point[1] * (b-t)) * this.canvas.height];
        }
        _rgba(value) { return 'rgba(' + [value[0]*255,value[1]*255,value[2]*255,value[3]].join(',') + ')'; }
        _paintScene() {
          const ctx = this.overlay.getContext('2d'); ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
          let scene; try { scene = JSON.parse(this.model.get('scene_json') || '{"groups":[]}'); } catch (_) { return; }
          const primitives = []; for (const group of (scene.groups || [])) for (const item of (group.primitives || [])) primitives.push(item);
          primitives.sort((a,b) => (a.zorder || 0) - (b.zorder || 0));
          const scale = (parseFloat(this.model.get('logical_dpi')) || 96) / 72 * (parseFloat(this.model.get('device_pixel_ratio')) || 1);
          for (const item of primitives) {
            const axis = this._axisFor(item.target); if (!axis) continue;
            ctx.save(); ctx.strokeStyle = this._rgba(item.color || item.facecolor); ctx.fillStyle = this._rgba(item.facecolor || item.color);
            if (item.type === 'line') {
              ctx.lineWidth = Math.max(1, (item.width_pt || 1) * scale);
              ctx.setLineDash(item.linestyle === '--' ? [6,4] : []); ctx.beginPath();
              item.points.forEach((p,i) => { const q=this._point(axis,p); if(i)ctx.lineTo(q[0],q[1]);else ctx.moveTo(q[0],q[1]); }); ctx.stroke();
            } else if (item.type === 'markers') {
              const radius = Math.max(2, (item.size_pt || 4) * scale / 2);
              for (const p of item.points) { const q=this._point(axis,p); ctx.beginPath(); ctx.arc(q[0],q[1],radius,0,Math.PI*2); ctx.fill(); if(item.edgecolor){ctx.strokeStyle=this._rgba(item.edgecolor);ctx.stroke();} }
            } else if (item.type === 'text') {
              const q=this._pointAxes(axis,item.position); ctx.font = (item.font_size_pt || 8) * scale + 'px ' + (item.font_family || 'sans-serif'); ctx.textAlign = item.horizontal_alignment || 'left'; ctx.textBaseline='middle'; ctx.fillText(item.text,q[0],q[1]);
            }
            ctx.restore();
          }
        }
        _event(e) {
          const rect=this.canvas.getBoundingClientRect(); const x=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)); const y=Math.max(0,Math.min(1,(e.clientY-rect.top)/rect.height));
          return {type:'pointer', action:e.action, x:x, y:y, button:e._zlcButton ?? null, double:(e.detail||0)>=2, step:e.step||0, key:e.key||null};
        }
        _send(e, force=false) {
          const now = performance.now();
          if (e.action === 'move' && !force && now - this._lastMoveSent < 30) return;
          if (e.action === 'move') this._lastMoveSent = now;
          this.model.send(this._event(e));
        }
        _bind() {
          this.canvas.addEventListener('contextmenu', e=>e.preventDefault());
          this.canvas.addEventListener('pointerdown', e=>{this._dragging=true;this._button=e.button+1;this.canvas.setPointerCapture(e.pointerId);e.action='press';e._zlcButton=this._button;this._send(e);});
          this.canvas.addEventListener('pointermove', e=>{if(!this._dragging)return;e.action='move';e._zlcButton=this._button;this._send(e);});
          this.canvas.addEventListener('pointerup', e=>{this._dragging=false;e.action='release';e._zlcButton=this._button;this._send(e,true);this._button=null;});
          this.canvas.addEventListener('pointercancel', e=>{this._dragging=false;e.action='cancel';e.button=null;this._send(e);});
          this.canvas.addEventListener('wheel', e=>{e.preventDefault();e.action='scroll';e.button=null;e.step=e.deltaY<0?1:-1;this._send(e);},{passive:false});
          this.el.tabIndex=0; this.el.addEventListener('keydown', e=>{if(e.key==='Escape'){e.action='key';e.button=null;this._send(e);}});
        }
      }
      return {ZlcRasterView: ZlcRasterView};
    });
  } catch (_) {}
})();
"""


class NotebookView:
    """A thin ipywidgets adapter over the same RasterPlotHost used by Qt."""

    def __init__(self, session: "PlotSession", *, close_session_on_close: bool = False) -> None:
        from .session import PlotSession

        if not isinstance(session, PlotSession):
            raise TypeError("session must be PlotSession")
        if not isinstance(close_session_on_close, bool):
            raise TypeError("close_session_on_close must be bool")
        self._session = session
        self._close_session = close_session_on_close
        self._host = RasterPlotHost.from_session(session, close_session=close_session_on_close)
        self._widget: Any | None = None
        self._output: Any | None = None
        self._front: RasterFront | None = None
        self._front_release: Any | None = None
        self._closed = False
        self._gesture_front: RasterFront | None = None
        self._gesture_axes: AxisTransform | None = None
        self._pointer_serial = 0
        self._scene_serial = -1
        self._kernel_lock = threading.RLock()

    @property
    def session(self) -> "PlotSession":
        return self._session

    @property
    def host(self) -> RasterPlotHost:
        return self._host

    @property
    def widget(self) -> Any:
        if self._widget is None:
            raise RuntimeError("call display() before accessing widget")
        return self._widget

    def _kernel_callback(self) -> Any:
        try:
            from IPython import get_ipython
            shell = get_ipython()
            loop = getattr(getattr(shell, "kernel", None), "io_loop", None)
            add_callback = getattr(loop, "add_callback", None)
        except (ImportError, AttributeError):
            add_callback = None
        if not callable(add_callback):
            return None
        return add_callback

    def _schedule(self, callback: Any) -> None:
        add_callback = self._kernel_callback()
        if callable(add_callback):
            add_callback(callback)
        else:
            callback()

    def _publish_front(self, front: RasterFront) -> None:
        with self._kernel_lock:
            if self._closed:
                return
            if self._front is not None and front.identity.sequence < self._front.identity.sequence:
                return
            self._front = front
            widget = self._widget
            dragging = self._gesture_front is not None
        if widget is None:
            return
        widget.width = front.buffer.width
        widget.height = front.buffer.height
        widget.logical_width = front.logical_size[0]
        widget.logical_height = front.logical_size[1]
        widget.logical_dpi = str(front.logical_dpi)
        widget.device_pixel_ratio = str(front.device_pixel_ratio)
        widget.rgba = front.buffer.pixels
        widget.axes_json = json.dumps([_axis_to_dict(axis) for axis in front.interaction.axes], separators=(",", ":"))
        if not dragging:
            widget.scene_json = json.dumps({"groups": []}, separators=(",", ":"))

    def _on_front(self, front: RasterFront) -> None:
        self._schedule(lambda: self._publish_front(front))

    def _axis_for(self, front: RasterFront, x: float, y: float) -> AxisTransform | None:
        for axis in front.interaction.axes:
            left, top, right, bottom = axis.bounds
            if left <= x <= right and top <= y <= bottom:
                return axis
        return None

    def _pointer_message(self, content: Mapping[str, object]) -> None:
        if self._closed or self._front is None:
            return
        self._pointer_serial += 1
        serial = self._pointer_serial
        action = str(content.get("action", "")).lower()
        x, y = float(content.get("x", 0.0)), float(content.get("y", 0.0))
        front = self._front
        if action == "press":
            self._gesture_front = front
            self._gesture_axes = self._axis_for(front, x, y)
        elif action in {"move", "release", "cancel"} and self._gesture_front is not None:
            front = self._gesture_front
        axes = self._gesture_axes if action in {"move", "release"} else self._axis_for(front, x, y)
        identity = front.identity if action != "cancel" else None
        interaction = front.interaction if action == "press" else None
        future = self._host._pointer_event(
            action,
            x,
            y,
            button=None if content.get("button") is None else int(content["button"]),
            double=bool(content.get("double", False)),
            step=float(content.get("step", 0.0)),
            key=None if content.get("key") is None else str(content["key"]),
            identity=identity,
            axes=axes,
            interaction=interaction,
        )
        future.add_done_callback(
            lambda completed: self._schedule(
                lambda: self._consume_pointer(completed, action, serial)
            )
        )

    def _consume_pointer(self, future: Future[Any], action: str, serial: int) -> None:
        if serial < self._scene_serial:
            return
        try:
            operation: RasterOperation[Any] = future.result()
        except Exception:
            if serial != self._pointer_serial:
                return
            self._scene_serial = serial
            self._gesture_front = None
            self._gesture_axes = None
            if self._host.front is not None:
                self._publish_front(self._host.front)
            return
        self._scene_serial = serial
        value = operation.value
        scene = getattr(value, "scene", None)
        if self._widget is not None:
            self._widget.scene_json = json.dumps(selector_scene_to_dict(scene), separators=(",", ":"))
        if operation.front is not None:
            self._publish_front(operation.front)
        if action in {"release", "cancel"}:
            self._gesture_front = None
            self._gesture_axes = None
            if self._widget is not None:
                self._widget.scene_json = json.dumps({"groups": []}, separators=(",", ":"))

    def _on_widget_message(self, _widget: Any, content: object, _buffers: object) -> None:
        if isinstance(content, Mapping) and content.get("type") == "pointer":
            self._pointer_message(content)

    def display(self) -> None:
        if self._closed:
            raise RuntimeError("NotebookView is closed")
        if self._widget is not None:
            return None
        global _WIDGET_MODULE_LOADED
        widget_class = _widget_class()
        try:
            import ipywidgets as widgets
            from IPython.display import Javascript, display
        except (ImportError, ModuleNotFoundError) as error:
            raise BackendUnavailableError(
                "NotebookView requires IPython and ipywidgets"
            ) from error
        if not _WIDGET_MODULE_LOADED:
            javascript = Javascript(_WIDGET_JS)
        else:
            javascript = None
        self._widget = widget_class()
        self._widget.on_msg(self._on_widget_message)
        self._front = self._host.wait_for_front()
        self._front_release = self._host.subscribe_front(self._on_front)
        self._publish_front(self._front)
        self._output = widgets.Output(
            layout=widgets.Layout(border="0", padding="0", margin="0")
        )
        display(self._output)
        with self._output:
            if javascript is not None:
                display(javascript)
                _WIDGET_MODULE_LOADED = True
            display(self._widget)
        return None

    def live_controller(self, contract: Any, *, refresh_interval_ms: int | None = None) -> Any:
        self._host.wait_for_front()
        return self._host.live_controller(contract, refresh_interval_ms=refresh_interval_ms)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._front_release is not None:
            self._front_release()
        self._front_release = None
        self._host.close()
        if self._widget is not None:
            close = getattr(self._widget, "close", None)
            if callable(close):
                close()
        self._widget = None
        if self._output is not None:
            clear = getattr(self._output, "clear_output", None)
            if callable(clear):
                clear(wait=True)
            close = getattr(self._output, "close", None)
            if callable(close):
                close()
        self._output = None

    def __enter__(self) -> "NotebookView":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


__all__ = ["NotebookView", "selector_scene_from_dict", "selector_scene_to_dict"]
