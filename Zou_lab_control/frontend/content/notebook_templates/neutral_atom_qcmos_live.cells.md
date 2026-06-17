<!-- cell:markdown -->
# qCMOS live 2D (相机单独连，不需要 sequencer)

最短路径：**只连 qCMOS 相机**，在 task console 里看实时 2D 图像。**sequencer 你自己连** —— 这个 notebook 完全不碰它（不 `na.connect` 整个实验、不开 RemoteSequencer）。

> ⚠️ **触发**：本框架的 qCMOS 适配器把相机设成**外触发**（`TRIGGERSOURCE.EXTERNAL`，上升沿）。所以每次 `acquire()` 是**等一个外部触发沿才返回一帧**。要看到图像，你那边的触发（sequencer 发到相机 / `ch11` 的脉冲）必须在跑；没有触发时 `acquire` 会等到 `timeout_ms` 然后抛 `TimeoutError`。下面把 `timeout_ms` 设成 2 秒，方便快速发现“没触发”。

<!-- cell:code -->
{{BOOTSTRAP_CELL}}

<!-- cell:code -->
import numpy as np

import Zou_lab_control.frontend as zf
from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera

zf.apply_style()

<!-- cell:markdown -->
## 1. 只连相机（不连 sequencer）

直接用框架的 `QCMOSCamera`，不经过 `na.connect`，所以不会去开 sequencer。下面这些参数就是 `Zou_lab_control/neutral_atom/configs/remote_template.json` 里 `camera` 那段的值，按你的相机改：`roi=(x, width, y, height)`，想要全幅就设 `roi=None`。

<!-- cell:code -->
cam = QCMOSCamera({
    "exposure": 0.02,             # 曝光，单位秒
    "readout_speed": 1,
    "roi": [1648, 64, 1144, 64],  # (x, width, y, height)；全幅用 None
    "device_index": 0,
    "timeout_ms": 2000,           # 等触发上限；没触发 2 秒就超时报错
})
cam.open()
cam.snapshot()

<!-- cell:markdown -->
## 2. 先抓一帧（确认相机 + 触发都通了）

`acquire(1)` 等**一个**外部触发沿、返回一帧。卡住约 2 秒后 `TimeoutError` = 没收到触发（去看你的 sequencer 触发是否在发、接线对不对）。

原始帧是 `(H, W)` 数组；框架的 2D 图按“每像素 `(x, y)` + 计数值”渲染（task console 的 2D 面板内部也是这么转的——**不能**把 `(H,W)` 直接丢给 `zf.plot(..., kind="2d")`，它要的是 `(N, 2)` 坐标）。下面把帧摊成坐标 + 一列值再画。

<!-- cell:code -->
frame = cam.acquire(1)[0]
print("frame:", frame.shape, frame.dtype, "min/max =", int(frame.min()), int(frame.max()))

ny, nx = frame.shape
xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
zf.plot(np.column_stack([xx.ravel(), yy.ravel()]), frame.ravel(),
        kind="2d", labels=("X (px)", "Y (px)", "counts"))

<!-- cell:markdown -->
## 3. 实时 2D：相机 measurement 节点 + task console

框架的 `CameraMeasurement`（相机 measurement 逻辑节点）：每个 shot 抓一帧、publish 成信号 `frame`。它的**数据源就是相机**——在看板里点这张 2D 面板的 **Edit…**，标签里的 **Acquisition** 段会列出相机的 `exposure` / `region`（ROI 端点），改了点 **Apply** 就**实时重配相机**（不用重开）。

控制台是**解耦**的：plot 面板是纯视图，只有**连了 signal 且产它的节点在跑**才显示数据。所以这张 2D 面板用 `source="frame"` 显式连到 `frame`，并把相机 measurement 节点交给 `running_nodes=`（开窗即自动 `start`）。

`%gui qt` 让 Jupyter 在 cell 之间替 Qt 窗口跑事件循环（看板的刷新 timer 才会动）；抓帧在节点的后台线程里，经线程安全的 `SignalHub` 交给看板。`rate_hz` 是抓帧节奏，真正多快还受触发频率 + 曝光限制。

<!-- cell:code -->
%gui qt

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

hub = SignalHub()
node = CameraMeasurement(hub, cam).start(rate_hz=4)

state = zf.TaskConsoleState(
    name="qcmos_live",
    panels=[zf.PanelConfig(kind="2d", title="qCMOS live", size="2x2", source="frame")],
)
console = zf.show_task_console(hub=hub, state=state, running_nodes=[node])
console

<!-- cell:markdown -->
## 4. 收尾

停相机 measurement 节点、关相机（关掉看板窗口本身也会停所有节点）。

<!-- cell:code -->
node.stop()
cam.close()
