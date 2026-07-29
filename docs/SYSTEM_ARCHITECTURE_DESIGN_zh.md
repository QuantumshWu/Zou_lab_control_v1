# Zou_lab_control 最终系统架构设计

## 1. 文档定位与权威

本文是仓库内唯一规范架构文档，定义最终产品语义、owner、依赖方向、公开合同、实现顺序与验收门。它不保存迁移轮次、旧实现说明、兼容策略或过程复盘。

权威顺序固定为：用户最新明确要求 → 当前完整 `/goal` → 物理与算法事实 → 本文 → 当前实现 → 仍有效的公开合同测试。若真实产品流、profiling、设备事实或代码依赖证伪本文，必须从物理语义和唯一 owner 重新推导，并在同一个 dependency-closed change 中同步修正文档、实现与当前测试；代码现状不能反向证明设计正确。

`AGENTS.md` 只规定执行、恢复、取证和提交方法；`docs/MAINTAINER_NOTES.md` 只记录当前 checkpoint。README、教程、测试名和注释都不是第二份架构权威。旧树只在调查一个明确旧行为或独立科学算法时作为定点 oracle，不能整体定义当前 UI、生命周期、包结构或运行模型。

全系统不可破的不变量：

1. 每个物理事实、数据语义、生命周期、持久格式和 presentation 决策只有一个 owner。
2. Dataset 永远是 `(R,P,*data_shape)`，标量永远是 `(R,P,1)`；R、P 与任意多维 trailing data 都不因显示、buffer 或 convenience 改写。
3. Task、Measurement、Processor、Fit、selector、Figure 和设备 capture 各守自己的语义边界；hosting/delivery policy 不产生第二套领域类型。
4. sample event、连续 monitor、有限 Dataset、formal association、artifact 与 GUI front 使用不同 typed contracts；latest/displayed 不能冒充物理 same-shot 或 scan authority。
5. GUI 线程只处理 O(1) 状态提交与不可变 front 安装；设备 I/O、数值求解、raster compose、selector materialization 和持久化均由明确 owner 托管。
6. 现有 RTL、bitstream 与 XDC 冻结；能由硬件确定的 pulse/trigger/exposure 时序继续由硬件执行。架构偏好不能授权重烧。
7. 不存在 alias、fallback、双 reader/writer、迁移 adapter、历史 archive、第二 renderer/form/codec、零消费者 abstraction 或为未来预建的工作流机器。
8. 不建立 caller-visible 软件内存预算、byte quota、pending/backlog 上限或预测性拒绝；真实分配失败和真实硬件容量各由其 owner 报告。
9. virtual 与 real 共享同一 application、Run、signal、Dataset、Figure 与 artifact 路径，只替换最低层设备 Port。
10. 完成的含义是正式产品流、故障反例、性能证据、真机资格和零残余同时成立；测试通过本身不是完成。

## 2. 唯一顶层架构

```text
foundation owners（彼此不反向认识产品）
  zlc_data       : Dataset/Point/CommittedTransform/Fit math values
  zlc_storage    : canonical bytes/CAS/atomic files/leases
  zlc_pulse      : pulse document/compiler/deployed transport facts -> frozen FPGA/RTL

zlc_frontend      -> zlc_data
  generic Figure/View/selector/Fit presentation + render

zlc_neutral_atom  -> zlc_data + zlc_storage + zlc_pulse
  devices/Logic-Node leaves/Run/SignalPlane/artifact lineage
  optional leaf ui submodule -> zlc_frontend（inert discovery不加载它）

zlc_workbench     -> zlc_frontend + zlc_neutral_atom（必要时消费zlc_pulse public facts）
  Qt product composition only

Zou_lab_control   -> 上述 public contracts
  public Experiment API + composition + application lifetime
```

禁止的反向边包括 data/frontend→neutral、pulse→neutral、frontend→Workbench、storage→任何实验域，以及 leaf ui→Workbench。跨 owner value 的编码必须调用 owner projector/parser，不复制字段表。

不可破的 owner 规则：

1. `zlc_data` 拥有数据结构和数值语义；它不知道设备、Logic Node、Qt 或 Matplotlib。
2. `zlc_frontend` 拥有所有通用 Figure/View/selector/Fit presentation、Divider、panel size、style、default resolver 与 renderer；leaf 只能构造领域 `FigureIntent`（source、plot intent、title/axis semantic labels），再调用同一个 frontend entry 编译 contract/display/session。TaskConsole、Calibration report、DataFigure、FigureViewer 均不得手写 composer、geometry、default view 或 style。
3. `zlc_neutral_atom` 拥有设备能力、领域 Logic Node、Run、signal transaction、artifact lineage；它不拥有通用 Figure，也不解析 pulse backend 私有字典。
4. `zlc_pulse` 拥有 pulse 文档、编译、部署几何、typed execution observation 和 remote transport；它不反向导入 neutral。
5. `zlc_workbench` 只拥有 Qt 窗口、卡片、路由、surface-scoped state、cancel/close 和产品布局；它通过 composition 注入的窄 compute submit/cancel/completion port 提交 Fit 等纯计算，不得重新定义领域字段、Dataset shape、Fit 算法、执行后端或 renderer。Qt callback 不得遍历、复制、编码或 hash 大 ndarray。
6. `Zou_lab_control` 保留为脚本、notebook 和 desktop 共用的稳定 API 与 composition root；它只做生命周期和窄委托，不实现领域算法或 Qt/Fit 状态机。
7. 新增或删除内建 Logic Node 的源码只改该叶包；fixed-namespace discovery 自动发现，不存在第二份中央 Logic Node installation 列表。只有新增真实物理 device/adapter instance 时才增加对应 device leaf 与部署配置。部署 allow/deny policy 可以选择已发现的包，但不是 concrete import 表，也不是新增 leaf 的必改源文件。

叶包边界也固定：默认表单、choice、API、resource 与 signal requirements 都在 inert descriptor 中声明；composition 在 Experiment 可用前一次解析成 frozen narrow facts。普通 UI 完全由通用 declaration projector 生成。确实无法由声明表达的可选 `ui/**` 只导出 lazy `UiContributionDescriptor(module, symbol)`；headless只验证它位于本leaf namespace且descriptor canonical，不加载Qt；Workbench product启动时在窗口可用前解析并类型校验factory，再通过frontend-owned generic UI context实例化。leaf UI不导入Workbench、catalog或service graph，失败则该product启动失败而不是静默回退。

## 3. 数据与 point domain 的唯一终态

### 3.1 物理 shape

```text
values.shape == (R, P, *data_shape)
scalar.shape == (R, P, 1)
```

- R 和 P 永远各是一个物理维。
- `data_shape` 可为任意多维，不能通过 `reshape(...)[0]`、flatten、singleton 或 rank 猜测丢失。
- 7×7×7 MOT scalar 的 canonical signal shape 只能显示为 `R × 343 × (1)`；`7 × 7 × 7` 只在独立的 GridTopology metadata/facet 控件显示，绝不混入 shape 字符串，更不是 `(R,7,7,7,1)`。

### 3.2 Validity

Validity 是 `zlc_data` 的 typed 数据事实，不能用 NaN、0、shape、rank 或广播巧合替代：

```text
ValueValidity = Valid | Invalid | ComponentValidity(axis_ids, mask)

DatasetValidity =
    Valid | Invalid
  | CellValidity(mask[R,P])
  | DatasetComponentValidity(axis_ids, mask[R,P,*declared_component_shape])
```

- `ComponentValidity`只属于单个 `Value`；`DatasetComponentValidity`只属于完整 Dataset。两者的 `axis_ids` 都必须是 trailing data axes 的有序子集，mask 只能按这些具名 axes 对齐/广播，调用者不得按 ndarray 尾部形状猜。
- scalar carrier仍有 trailing `(1)`，但其 validity不因此成为 `(R,P)` cell validity；carrier、cell与component是三个不同语义层。
- producer/processor的schema在generation建立时声明 validity contract；从VALUE变成component mask、component axes改变或crop使output schema改变都必须建立新generation，不能根据首帧动态改合同。
- Dataset builder、monitor ingest与任何 data patch 都在同一revision事务内原子提交values、validity、event/provenance和coverage；消费者不能看到新values配旧mask。
- Selection 同步裁剪mask；Reduction按显式`ALL_REQUIRED/ANY_VALID/MIN_COUNT(n)`等policy计算输出validity；Fit逐batch排除无效observations并保留per-cell failure；Histogram记录dropped count；Meter/renderer显示invalid而不回退其它component。领域算法若有更强物理规则，由该leaf在通用validity之上声明，不能绕过它。
- 完整image可以使用uniform/packed/broadcast representation避免复制同尺寸bool数组，但该优化不能改变具名axis语义，也不能把source dtype转为float。

### 3.3 PointTable

Dataset 的 point truth 固定为：

```text
PointColumn:
  coordinate_id: AxisId
  name: str
  role: AxisRoleId
  value_kind: NUMERIC | TEXT
  values: immutable tuple of exactly P canonical scalar values
  unit: canonical str | None
  coordinate_frame: canonical str | None       # opaque equality tag，不做坐标代数

PointTable:
  row_count: P >= 1
  columns: tuple[PointColumn, ...]
  implicit row identity: zero-based ordinal 0 .. P-1
```

确定规则：

- row ordinal 是 frozen Dataset/Run 内唯一 point identity；Dataset/Run generation 提供作用域，不增加永远等于 ordinal 的 UUID `PointId`。
- `zlc_data.canonical_coordinate_scalar` 是 AxisSpec、PointColumn 与 GridTopology coordinate domain 共用的唯一 normalizer/equality owner：vocabulary 为 `None | str | int | finite float`，NumPy scalar 先归一化，integral float 归一为 int，`-0.0` 归一为 `0.0`；bool、NaN/Inf、bytes、容器和任意 Python object 一律拒绝。codec 使用同一 projector，禁止三处各写一套规则。
- PointColumn 的 `value_kind` 纳入 codec/fingerprint并显式声明 NUMERIC 或 TEXT；除 `None` 外不得混合数值与字符串，TEXT 不得有 unit。`None` 是 missing coordinate，不是一个可 facet/group 的类别：用到该 coordinate 的 display 会把对应 row 标为 invalid并显示 dropped count，Fit 排除并记录 observation count，GridTopology domain/映射则直接拒绝 `None`。
- `AxisRoleId` 在这里只是语义标签，不把 PointColumn 变成独立 ndarray axis。合法 point-domain role 明确限定为 `SCAN_POINT/READOUT_EVENT/MONITOR_HISTORY/SPATIAL_X/SPATIAL_Y/SPECTRAL/HISTOGRAM_BIN/SITE`；`REPEAT/SCALAR/COMPONENT` 不得进入 PointColumn。未来若新增合法 role，必须先在唯一数据合同中扩展该闭集及 codec test，不能由 leaf 自行放宽。
- `coordinate_id` 复用已有 `AxisId`，不再增加只在 point column 中改名转发的 `PointCoordinateId`。它在一个 PointTable 内唯一；每列长度必须等于 P；PointTable、GridTopology 和 cell schema 全部进入 DatasetSchema fingerprint。
- coordinate column 和完整 coordinate tuple 都允许重复；coordinate equality 不是身份。
- 行顺序就是 authored、compiled、emitted、captured 和 provenance 顺序。
- `P=1` 且零 coordinate column 完全合法。
- 同一 Dataset 的每个 R 使用同一个 PointTable；若不同 repeat 使用不同 point sequence，则必须新 generation，或表达成 R=1 的 event dataset，不能假装同一 schema。

### 3.4 可选 GridTopology

```text
GridTopology:
  dimension_ids: ordered tuple[AxisId, ...]
  coordinate_domains: one ordered unique scalar tuple per dimension
  row_to_cell: exactly P entries, ordinal -> logical cell tuple
```

规则：

- 只有知道数据确实是 grid 的 producer 才能声明；绝不从 rank、unique-count、Python 变量或笛卡尔积推断。
- `row_to_cell` 必须 in-bounds 且 injective；可以 incomplete，因此支持 sparse 和 serpentine。completeness 是派生事实，不是另一个布尔真相。
- 每个 mapped row 的 PointColumn 值必须等于相应 dimension domain 的 cell 值；每个 dimension_id 必须引用唯一 PointColumn 并只能出现一次，它同时就是该 logical dimension 的身份。不得为同一 tuple entry 再建 `GridDimension` 或 `GridDimensionId` wrapper。
- 同一 sweep 内重复访问同一 logical cell 的 trajectory 不带 GridTopology；完整 grid 的重复 sweep 应进入 R。
- 只有把 point rows 映射成二维/多维 logical cells（densify、grid image、topology-dimension selector/facet）才需要 GridTopology；普通 `PlotKind.GRID` 是“一个 FACET source 的 small multiples”容器，可 facet R、trailing data axis、PointRows（逐ordinal）或 PointCoordinate，完全不要求 GridTopology。
- `AxisLayout` 仍可服务 FitResultBatch 等真正具有独立 batch grid 的结果，但不再充当 Dataset point truth。

### 3.5 Point 与 View/Fit 的连接

`ViewSpec` 不再并存“tensor axis bindings”和另一套残缺的 point projection。它只保存一套 closed typed source bindings：

```text
AxisSourceRef:
  kind: TENSOR | POINT_ROWS | POINT_ORDINAL |
        POINT_COORDINATE | GRID_DIMENSION
  axis_id: AxisId | None

SourceViewBinding:
  source: AxisSourceRef
  role: X | IMAGE_X | IMAGE_Y | SAMPLE | BATCH | FACET |
        SELECTED | REDUCED
  selector/reduction: 仅在该 role 要求时存在

ViewSpec:
  schema_fingerprint
  intent
  source_bindings
  point_ordinals: tuple[int, ...] | None       # 唯一 P-row filter authority
```

`AxisSourceRef` 是一个 discriminated immutable value，不是五个 subclass、五套 descriptor 或五个模块。`TENSOR`、`POINT_COORDINATE`、`GRID_DIMENSION` 携一个已有 `AxisId`；`POINT_ROWS`、`POINT_ORDINAL` 不携 id。`point_ordinals=None` 表示全部 authored rows；显式值必须非空、严格递增、唯一且在表内。不得再建立 `PointCoordinateId`、`PointRowSelection`、`PointGroupSpec` 或同义 request DTO。

精确验证规则：

- 合法 role 闭集固定如下；不在表中的组合在 bind 时拒绝，UI 不显示：

| source | 合法 display roles | 额外条件 |
|---|---|---|
| `TENSOR` | X、IMAGE_X/Y、SAMPLE、BATCH、FACET、SELECTED、REDUCED | 由 ViewContract/AxisRole 再收窄；`LatestNonempty` 只允许 repeat source |
| `POINT_ROWS` | SAMPLE、BATCH、FACET、REDUCED | 先应用唯一 `point_ordinals`；BATCH/FACET 表示每个 surviving row 一个 cell；SAMPLE 只在 Histogram；binding 不得再携 row selector |
| `POINT_ORDINAL` | X；以及 PRESERVE_ROWS 下与一个 TENSOR source 配对的 IMAGE_X 或 IMAGE_Y | 唯一 `0..P-1` 坐标；不得 BATCH/FACET。Image 中它定义 dense row-cell geometry |
| NUMERIC `POINT_COORDINATE` | X、BATCH、FACET；以及 PRESERVE_ROWS 下与一个 TENSOR source 配对的 IMAGE_X 或 IMAGE_Y | BATCH/FACET 按 canonical value 分组；Image 仍按 authored row ordinal 放置 cell，只把 coordinate 作逐 row label/metadata，不按不规则值拉伸/重采样；不可 SAMPLE/REDUCED |
| TEXT `POINT_COORDINATE` | BATCH、FACET | 不得作为数值 X/Image/Fit independent source；missing rows 按固定规则丢弃并计数 |
| `GRID_DIMENSION` | X、IMAGE_X/Y、BATCH、FACET、SELECTED、REDUCED | 必须 `GRID_CELLS` 且 topology member；不可 SAMPLE；有序 categorical domain 由 cell ordinal 渲染并显示原 label |

- data-owned `zlc_data.resolve_point_rows(PointTable, GridTopology|None, point_ordinals, group_sources)` 只接受普通 `tuple[AxisSourceRef, ...]`，不接受 request class。group sources 只允许单独的 `POINT_ROWS`，或一组 `POINT_COORDINATE`，或一组 `GRID_DIMENSION`；raw/topology不能混。resolver把raw sources按PointTable column declaration、topology sources按GridTopology dimension declaration canonicalize，caller/wire点击顺序不是authority。它输出唯一 immutable `ResolvedPointRows`：surviving ordinals、每个 group 的 typed sources/address/canonical value tuple、exact member ordinals与missing/drop count。每个point ordinal本身就是唯一observation address，禁止再存一个镜像字段或wrapper。PointRows groups按authored ordinal；raw composite key groups按首次出现；topology composite address按declared domains的lexicographic cell order且只返回实际occupied groups；组内永远authored ordinal order。FitResultBatch durable descriptor保存typed sources、group address/value与exact ordinals，不能只保存ref/string。
- point 解析分成两层但不产生第二结果模型。上述 data resolver 只认识 row、coordinate、filter/group，绝不认识 X/FACET/PlotKind。frontend 在现有 Figure contract owner 中用私有函数验证 ViewContract/source-role 组合，随后直接委托 data resolver并消费同一个 `ResolvedPointRows`；少量 metadata/cardinality 只用私有函数局部 tuple。不得新增 frontend resolved DTO、resolver module或公开 projection descriptor。Curve/Image/Histogram/Grid evaluator 与 atomic-front→Fit translator 走同一私有入口；translator 再冻结为 data-owned CommittedTransform/FitSpec，禁止 zlc_data 反向导入 frontend。
- 组合合法性固定：`point_ordinals` 是唯一 row subset authority；`POINT_ROWS` binding 再携 selector、`POINT_ORDINAL` 做 BATCH/FACET、或任何独立 `SLIDER` role 都 bind-time 拒绝。`SLIDER` 只是 frontend 根据 `SELECTED + SelectorSpec`、source cardinality 与 editability policy投影出的控件，不进 ViewSpec/codec。同一个 underlying AxisId 不得同时以 `POINT_COORDINATE` 和 `GRID_DIMENSION` 绑定；一个 View 最多一个 `POINT_ROWS` consuming role；同一 source 只能出现一次；point-domain 总 facet 仍最多一个。`coordinate X + coordinate BATCH/FACET`、`POINT_ROWS SAMPLE + raw POINT_COORDINATE BATCH/FACET`、`one numeric point coordinate/ordinal image axis + one tensor image axis` 合法；`POINT_ROWS REDUCED/BATCH/FACET + coordinate X`、`POINT_ROWS SAMPLE + GRID_DIMENSION BATCH/FACET`、两个 raw point coordinates 组成 IMAGE_X/Y、raw coordinate binding 与任何 grid source 混用均拒绝。grid source出现后，所有 point coordinate display/group roles 都必须走 topology source。
- 五种 source kind 具有不同语义，但后四种全部共享一个物理 PointRows domain，绝不能因为出现两列坐标就做 Cartesian product。
- coordinate Eq/Range/In只可作为UI临时query；Eq/In的literal必须与column value_kind匹配，Range只允许NUMERIC，None永不匹配。Apply时必须在frozen PointTable上解析成exact `point_ordinals`才进View/Fit/codec；重复coordinate命中全部rows，selection按authored ordinal保留不重排，空结果明确NEEDS_INPUT/validation error而不是伪造空Dataset。
- `POINT_ORDINAL` 的 metadata 唯一定义为 `name='point'`、`role=SCAN_POINT`、`unit=None`、`coordinate_frame=None`、values=`0..P-1`；View/Fit/evaluator 只能读取同一 helper 的结果，不能各自命名或补 descriptor。
- point-domain mode 不另存字段：没有`GRID_DIMENSION` source时派生`PRESERVE_ROWS`，P是一条dense authored sequence；出现任一`GRID_DIMENSION` source时派生`GRID_CELLS`。`POINT_COORDINATE`可作Curve真实X；`POINT_COORDINATE/POINT_ORDINAL`也可与恰好一个`TENSOR` source组成`P × data-axis`Image，但其pixel geometry永远是authored ordinal cells，raw coordinate仅作labels/metadata，故duplicate/nonmonotonic/irregular值不被伪装成均匀物理距离。两个相关point columns绝不能分别当IMAGE_X/IMAGE_Y。
- `POINT_COORDINATE(BATCH/FACET)` 只表示按 exact canonical composite value tuple 对 rows 分组，不是独立 tensor batch axis：组内 row 保持 authored order，组顺序按首次出现；多个 group coordinates 的 source/address 顺序按 PointTable column declaration，不按hash、名称、wire点击或序列化顺序。Point reduce/sample作用于`POINT_ROWS`，row select只在`point_ordinals`，不作用于coordinate field。
- `POINT_ROWS(SAMPLE)` 明确把 surviving P rows 当样本；因此 Calibration 可以同时把 R 与 P 设为 SAMPLE。`POINT_ROWS(REDUCED/BATCH/FACET)` 分别表示沿 surviving P rows reduce、逐 row batch 或逐 row small multiples；选 row 只由 `point_ordinals` 表达，不伪造 AxisSpec 或第二 selector。
- 派生的 `GRID_CELLS` 只有 GridTopology 存在且 source-coordinate/domain/row mapping 验证通过才合法；`GRID_DIMENSION` 可做 X/IMAGE_X/IMAGE_Y/SELECTED/REDUCED/BATCH/FACET。二维 scalar grid image 必须绑定两个 grid sources；sparse cell 保持 invalid mask，不填值、不平均。
- 一个 `PlotKind.GRID` 恰有一个 FACET binding；facet 可以来自任意合法 source。仅当该 source kind 是 `GRID_DIMENSION` 时要求 topology。
- Curve/Histogram/Image evaluator 消费同一 SourceViewBinding 语义。trailing spatial data axes 可直接形成 Image；普通 finite 非网格多维 scan、相关 `(x_i,y_i,z_i)` trajectory、重复点和 hysteresis 由 PointTable 原样支持。
- adaptive/growing scan table 不属于当前 frozen finite baseline：bind 后增长明确拒绝，不隐含支持。

`FitSpec` 复用同一个 source vocabulary，但不是 ViewSpec 的别名。所有 source 都只相对同一个 `CommittedTransform.effective_output_schema` 解释；因此 Histogram 等 transform 生成的 axis 仍直接使用 `AxisSourceRef.tensor(axis_id)`，不得再增加一个重复携带 transform digest 的 wrapper：

```text
CommittedTransform:
  source_schema_fingerprint
  exact_point_ordinals                     # 唯一 point-row selection payload
  ordered typed operations                 # Tensor SELECTED/REDUCED/SAMPLE；
                                           # GridDimension SELECTED/REDUCED；
                                           # PointRows SAMPLE/REDUCED；payload只在这里一份
  actual transform parameters/outputs      # 例如 actual histogram bins/effective axes
  effective_output_schema + digest

FitSpec:
  committed_transform
  independent_sources: ordered tuple[AxisSourceRef, ...]  # model arity order
  batch_sources
  model + arguments
```

- 多元模型可显式选择多个 `POINT_COORDINATE` source；它们共享同一批 rows，不产生 P²/P³。
- Fit independent source 只允许 effective schema 中的 numeric `TENSOR`、`POINT_ORDINAL`、NUMERIC `POINT_COORDINATE` 或 NUMERIC `GRID_DIMENSION`；`POINT_ROWS`、TEXT/missing coordinate 不能当模型自变量。ordered tuple 严格保持 model arity 顺序，绝不 canonical-sort。
- Histogram Fit 冻结 exact sample projection 与 HistogramSpec/actual bins 为唯一 CommittedTransform；X 明确使用相对其 effective schema 的 `AxisSourceRef.tensor(HISTOGRAM_BIN axis id)`，不能按名字、renderer 当前 bins 或 input schema 猜。FitSpec 已内嵌该 transform，故 source 不重复保存 digest。
- Fit authority 只存一份：point-row filter 只在 `CommittedTransform.exact_point_ordinals`；tensor/grid selector、reducer 与 sample operation 只在它的 typed ordered operations；`CommittedTransform` 不再嵌套第二份 ViewSpec 或 row-selection object。`FitSpec` 只增加 model arity 与 batch grouping，不复制 transform payload。codec/bind 对重复或矛盾声明直接拒绝。
- Fit role 闭集固定：`batch_sources`只允许`TENSOR`、`POINT_ROWS`（逐surviving row）、`POINT_COORDINATE`（按值分组）或`GRID_DIMENSION`。transform中TENSOR可SELECTED/REDUCED/SAMPLE，GRID_DIMENSION只可SELECTED/REDUCED，POINT_ROWS只可SAMPLE/REDUCED；POINT_ROWS SELECTED永远非法，P-row filter只在exact_point_ordinals。一个source不能同时属于independent/batch/sample/reduction/selection，raw/topology coordinate也不能双绑；POINT_ROWS SAMPLE/REDUCED与raw POINT_COORDINATE independent互斥，Histogram X只能引用同一 committed effective schema 中的HISTOGRAM_BIN tensor source。Fit bind只调用data-owned resolve_point_rows；来自Figure的Fit由frontend私有resolver翻译成该data plan。
- 未选择的 coordinate columns 只是 metadata，绝不自动变成 batch。用户选择 coordinate 为 group/batch 或 GridDimension 为 BATCH 时，Fit commit 冻结 exact group membership、batch address 和 row ordinals；重复值可包含多个 rows，稀疏组合写入 sparse batch layout。FitResultBatch 的 batch descriptor 保存 typed AxisSourceRef，而不是退化成 AxisId/string。
- Fit translator 可由当前可见 View 预填，但点击时必须重新验证并冻结上述独立 authority；View 的动态 selector、latest 或 display reduction 不能直接进入 solver。

## 4. Repeat、View、Fit 的唯一终态

四个同名但正交的概念必须分开：

1. Pulse `RepeatRegion`：单个 point 内由 FPGA 执行的 timeline loop。
2. acquisition/scan sweep count：完整 point table 获取多少遍；只有它形成 Dataset R。
3. Setting repeat presentation：只改变 R 在当前 Figure 中如何显示。
4. monitor rolling history：设备/monitor 的有限历史，不是 Dataset R，也不进入公开 Camera signal shape。

`ViewSpec` 是 repeat presentation 的唯一保存真相，UI option 直接携带 canonical binding action：

| UI 语义 | canonical binding | 合法范围 |
|---|---|---|
| Mean | `REDUCED + MEAN` | Curve/Image/Histogram/Meter |
| Sum | `REDUCED + SUM` | Curve/Image/Histogram/Meter |
| Latest | `SELECTED + LatestNonempty` | Curve/Image/Histogram/Meter |
| Index N | `SELECTED + FixedIndex(N)` | Curve/Image/Histogram/Meter |
| Overlay | `BATCH` | Curve/Histogram |
| Pool as samples | `SAMPLE` | 只在 Histogram View；Fit 通过冻结该 Histogram sample projection 获得 authority |
| Facet | `SourceViewBinding(..., FACET)` | 通用 Grid facet selector，可选择 R、真实 data axis、PointRows/PointCoordinate 或 topology dimension；不属于 repeat 下拉 |

一个 Grid view 总共只能有一个 facet source。UI 可把所有合法项放在同一 selector，但 item data 必须保存 typed `AxisSourceRef`，不能把 `AxisId`、`POINT_COORDINATE` 与 `GRID_DIMENSION` source 混成字符串；后二者即便携同一 AxisId，语义也不同。

该收敛不删除产品默认。`zlc_frontend.default_repeat_binding(intent)` 是唯一纯策略函数，直接返回 canonical binding：Curve/Image=`REDUCED+MEAN`，Histogram/Distribution=`SAMPLE`，Meter=`SELECTED+LatestNonempty`；Grid 若 facet source 是 R 则 R=`FACET`，否则继承 cell intent 的默认。UI、Calibration 和 FigureViewer 不得重写这张表。

`AUTO` 只是一种 resolver 状态，永远不是数值 SpinBox 中的字符串值。

默认显示由 `zlc_frontend` 唯一的 `ViewContract + default_view(schema, plot_kind)` 纯函数生成：底层数据非破坏，display-only reduction 必须由 role/contract 明确允许、永久可见且可改；PanelCard、Calibration、FigureViewer 和 leaf 只能请求或消费该结果，不能各自再选默认轴。禁止按 shape/rank/singleton/data value 猜。合同可唯一决定时直接 `RESOLVED`；否则 `NEEDS_INPUT`。删除无人消费且会过期的 alternatives snapshot 与 `REVIEW_REQUIRED`，合法 choices 从当前 schema+contract 实时查询。

`ViewContract`本身是frontend-owned immutable policy value，明确列出每个display role的allowed source kinds、ordered preferred AxisRoleIds、explicit sample sources、default-batch sources与facet candidates；leaf只能选标准contract或在FigureIntent中显式绑定source，不能提供callback。`default_view`只选“最高优先级层恰有一个candidate”的项；同层多个candidate必为NEEDS_INPUT，不按axis长度/名称/数据值破平局。

默认 policy 是可执行闭表，不留给各 plot kind 猜：

| plot intent | 必须解析的 source | 其它有信息 source 的默认 |
|---|---|---|
| Curve | explicit X优先；否则唯一role-preferred numeric X；普通point sequence无唯一coordinate时使用`POINT_ORDINAL` source；多个preferred candidates则NEEDS_INPUT | R=MEAN；非R source仅在contract明确`default_batch`时BATCH，否则可index者`SELECTED+FixedIndex(0)`并永久显示；绝不默认reduce；不可选择则NEEDS_INPUT |
| Image/SiteMap | 恰好两个合法IMAGE_X/Y；只允许两个TENSOR sources、两个GRID_DIMENSION sources，或一个numeric POINT_COORDINATE/POINT_ORDINAL source + 一个TENSOR source | R=MEAN；其它非R source一律可index则`SELECTED+FixedIndex(0)`并显示，不默认reduce；不可选择则NEEDS_INPUT |
| Histogram/Distribution | SAMPLE sources必须由FigureIntent/ViewContract role policy明示；R默认SAMPLE，P只有contract明示才用PointRows SAMPLE | 非sample source仅在contract明确`default_batch`时BATCH，否则可index者SELECTED；FACET只由外层Grid拥有；自动bimodal analysis不改变authority |
| Grid | explicit FACET优先，否则唯一facet-preferred source；再加typed cell intent（Curve/Histogram/Image） | 恰好一个facet；移除后递归使用cell policy；无/多候选则NEEDS_INPUT；GridTopology只在选GRID_DIMENSION source时需要 |
| Rolling | MONITOR_HISTORY 是唯一 X | 其余按 Curve 规则；rolling history 不是 R |
| Meter | 仅内部 scalar primitive，R=LatestNonempty | 任一未消费的有信息 source 都使 contract NEEDS_INPUT |
| Pulse | 专用 pulse FigureIntent | 只消费 pulse document/timing contract，不参与 Dataset 轴猜测 |

“自动”只生成一份可见、可改、display-only 的 ViewSpec；`FixedIndex(0)` 等默认选择必须在 Setting 中显示。它从 declared role 与 schema order 得出，绝不从 rank/singleton/data value 推断，也绝不把 display 默认直接升级为 Fit/Scan authority。

显示和权威之间是类型边界。`zlc_frontend.FigureFront`是一次immutable atomic presentation front，明确包含：base 的 exact DatasetRevisionRef、ViewSpec、frontend-owned `EvaluatedProjectionFront`（source/effective schema、resolved row/group memberships与observation addresses、actual histogram bins/effective axes、projection digest；不复制整个source ndarray）、`PlotDisplayState`（viewport/color range/current facet等size/DPR-independent visual state）、`RenderedGeometry`（logical size、DPR、raster pixel size、每panel Divider data box与由数值/enum描述的双向coordinate transform；不得含QWidget/Artist/callable）、`SelectionState`、canonical base raster，以及可选的 typed overlay layer。RenderedGeometry与base raster必须同一compose原子产生；selector/hit-test始终绑定base ref。Fit overlay另带自己的exact source ref、projection digest、兼容性和`CURRENT/LAGGING/INCOMPATIBLE`状态：同generation/schema/axes时可在最新base上继续显示但必须可见标出lag，语义不兼容则隐藏；不得伪装成与base同shot。live Qt只用同一frontend overlay geometry/style经轻量painter/blit更新，headless export由同一overlay contract合成，Fit完成不得因此重画Matplotlib base。`ViewSpec`只含数据投影，viewport/color range不得进入Fit，current facet/explicit selector只有点击Fit时才被translator明确提升。FigureFront不导入neutral transaction。

neutral signal generation只冻结自己的route identity、schema、cadence与parent graph。一个UI attachment只有在composition已纯验证并冻结对应的generation-static `FigureIntent` 后才进入UI topology；这一步不产生per-revision presentation对象，失败也不阻止neutral signal。每个revision的唯一权威仍是neutral `SignalFront`；`zlc_workbench`从同一个exact front与attachment intent生成完整`FigureFront`。同一requested causal cut的全部panel先全部生成成功，再原子present一个完整board front；任一attachment投影失败时保留上一完整board并显示typed nonfatal attachment错误，不能逐panel错shot推进，也不能杀死GUI timer。panel私有值只是该board中的`(SignalPublication, FigureFront)`。Fit/gesture等显式operation在一个freeze边界直接强引用这两个既有immutable对象；publication本身携exact transaction/provenance并必须已在发布边界成为进程自有immutable data，因此无需通用borrow/lease/release manager，也绝不按name/latest重建。standalone artifact Figure只持纯FigureFront，仍可画selector/Fit overlay并返回本地FitResult，但没有parent时不得伪造或发布derived signal。translator只消费该frozen FigureFront，selector/Fit completion只消费同一私有operation record中的exact publication，绝不能分别读取两个latest：

- Scan 永远不继承 display ViewSpec。
- 点击 Fit 是一次明确确认动作；它可读取当前可见语义来预填，然后立刻冻结独立 `FitSpec/CommittedTransform`。
- `LatestNonempty` 在该 frozen front 上解析为“所选 view 至少一个 component valid”的最大单一 R index；整 slice 使用同一 index并保留 component validity。`SELECTED+SelectorSpec` 冻结 exact index；UI slider不是语义字段。point row filter冻结exact ordinals；mean/sum冻结`ReductionSpec`；explicit Area/ROI冻结exact selection；Histogram SAMPLE冻结exact sources/bins；BATCH/FACET冻结Fit batch/group membership。
- View role `SAMPLE` 只在 Histogram 合法；“Fit sample”是 translator 把当前 Histogram 的 exact sample projection 写进 FitSpec，不允许非 Histogram View 静默开放 SAMPLE。
- Fit绝不隐式Hold、Pause或钉住live base。Monitor/Setting激活surface-local live/latest语义：base持续呈现每个最新front；每surface最多一个active candidate和一个latest-pending完整candidate，新front只替换尚未开始的pending并让旧私有强引用自然释放，不反压producer、不混revision，完成后公平调度最新pending。结果保留exact source/projection provenance，参数以独立`EVENT_RESULT` atomic sibling bundle发布，diagnostics只留FitResult/UI/artifact，不进signal plane，也不发布拟合曲线数组；overlay按上一段的current/lag/incompatible规则独立呈现。source generation/schema/axes/CommittedTransform/model/facet改变时取消该surface状态或使旧overlay incompatible。live/latest只是scope语义，不授权新增public session类。
- Edit/DataFigure Fit只是一种属于该snapshot surface的一次性语义；它可以冻结自己的source，但不得读写Monitor surface的Fit spec/source/result、不得改变任何Monitor cadence，也不得进入continuous causal frontier。signal-connected Edit拥有exact publication时仍可把named params发布为独立EVENT_RESULT；standalone artifact无parent时只返回本地结果。显式Pause、press→release手势和Edit snapshot可以Hold自己的base；Fit按钮本身不能。live/snapshot是必须区分的scope，不授权新增public wrapper class；最小私有record守住source/cancel/result边界即可。
- Fit compute、Matplotlib base render和selector Dataset materialization在调度上不得相互阻塞，但不为三者各建public lane。composition只提供一个窄compute submit/cancel/completion seam并复用现有Matplotlib serial render owner；one-active+latest、公平与cancel是Workbench surface私有状态。Qt只O(1)提交immutable ref/spec并接收小结果/overlay primitives。执行后端不预先锁死为thread或process：先消除Python逐元素循环、全图坐标/validity/materialization与Qt侧codec/hash，再以真实profile决定；不得用独立进程掩盖可避免的复制，也不得引入caller-visible memory budget。

Selector也只有一个frontend contract/interaction engine，Curve/Histogram/Image/SiteMap/Grid cell全部复用FigureFront中同一RenderedGeometry/coordinate mapper、Divider与overlay painter，不得每kind自写。Cross发布exact rendered sample/cell的数据值（位置只进provenance，不另发XY signal）；Area发布一个保留source dtype/未选axes/component validity的selected Dataset，并把exact typed bounds/SelectionSpec放在同一signal的schema/provenance中，范围与数据同一authority，不再造第二个lossy bounds array；Grid委托当前cell并携facet address；不发布hover。press到release锁定同一immutable SignalPublication+FigureFront+RenderedGeometry；new source front可暂存到release后present，但resize/DPR导致geometry invalidation时必须原子cancel gesture且不发布final，再用新geometry compose，绝不用新widget geometry解释旧raster。drag中overlay与monitor-derived值连续更新，overlay用轻量painter/blit，不以每个pointer event整图compose。gesture active输出无FormalAssociationCapability；release冻结selection成新producer generation，之后对每个新parent做one-to-one projection并可获capability。Area crop shape、Fit parameter schema等改变必须换signal generation；monitor panel可重绑，已bind formal consumer遇generation change则失败。

## 5. Signal transaction 的唯一终态

```text
ProducerTransactionId:
  producer identity + generation + monotonic publication sequence

SignalPublication:
  one immutable transaction id
  one atomic sibling-signal bundle
  exact source Dataset revision/provenance

DerivedPublication:
  own transaction id
  exact parent transaction id(s)

SignalFront:
  immutable set of mutually coherent SignalPublications
  explicit transaction-graph frontier, never name-based reconstruction
```

- producer 在发布时创建 transaction；retainer 只管理生命周期，不能事后按 signal name 猜 transaction。
- selector、Fit 和 processor 派生信号必须携带 exact parent transaction。
- neutral SignalPlane 是唯一 transaction mint/retain/publish owner，并暴露窄 `DerivedSignalPublisher` port；frontend 完全不知道 ProducerTransactionId/OwnedSignalPublication。active producer/processor generation在bind时冻结typed parent graph、expected sibling bundle、failure policy与promotion policy，cycle启动即拒绝。`COHERENT_CONTINUOUS`只用于每个parent都应产出的processor/已commit selector；user-triggered Fit等`EVENT_RESULT`保留exact parent但从不作为source frontier的必到成员。Workbench board只提交当前实际连接的continuous signal-id immutable set；“causal cut request”只是该语义的名字，不授权public class。neutral只取这些signals的parent closure，不让未显示旁支或event result阻塞。
- neutral 为该closure生成immutable `SignalFront`；它就是被提升的causal cut：source与所请求derived outputs到达同一frontier后才原子提升；pending时继续呈现上一完整front；atomic siblings永不拆开；无依赖producer独立前进。formal/exact链每个transaction都处理，gap/failure fatal；monitor latest链可以按声明supersede整个未完成candidate，但只能丢弃整组、绝不能把不同parent的部分拼起来。Workbench所有相关panels从同一个promoted front取值，因此不会出现raw N与ROI/processor N-1；derived失败时保留上一完整front并显式呈现该generation error。
- UI attachment进入topology前先纯验证并冻结generation-static frontend FigureIntent；运行中每个revision不再prepare或配对presentation sidecar。Workbench只从neutral提升的同一个exact SignalFront生成全部FigureFront并原子安装完整board front；pointer gesture从press到release固定对应immutable publication和front。Area/Cross等持续route各自发布COHERENT_CONTINUOUS，Fit参数单独发布EVENT_RESULT，不能把它们塞进一个动态“panel outputs”bundle使任一事件阻塞continuous front。异步Fit用私有operation record强引用base front与exact SignalPublication；solver完成后通过DerivedSignalPublisher发布该publication的参数bundle，overlay source/lag按§4独立判断。
- `EmissionSlotId`（nonempty canonical string typed id）与`PointEmissionMap`都由`zlc_data`拥有，neutral只消费。formal finite consumer的publication cardinality必须在bind/preflight冻结：`ExactlyOnePublication`明确表示每个`(repeat, base_point_ordinal)`恰好一个atomic publication；`FixedPublicationEvents(ordered tuple[EmissionSlotId,...])`表示每个base cell恰好K>=2个有序publications（K=1 canonicalize为ExactlyOne），slot ids唯一。后者由collector在reservation前生成`PointEmissionMap(base_point_count, ordered slots, expanded_row -> base_ordinal+slot_ordinal)`，把base PointTable每行按slot顺序复制，并追加一个由consumer-binding id派生、冲突即拒绝、role=`READOUT_EVENT`、TEXT的emission-slot PointColumn。若base有GridTopology，则expanded topology在原dimensions后追加同一emission-slot GridDimension，row_to_cell=`base_cell + slot_ordinal`，保持injective；不得把K rows重复映射到同一base cell。expanded table/topology/map全部进schema/artifact fingerprint；runtime不得观察到K后再改schema。
- fixed-event collector 精确 reserve `R*baseP*K`，按 `(repeat, base_ordinal, slot_ordinal)` 消费；missing、surplus、错序、generation change 全 fatal。一次 SignalPublication bundle 内的多个 sibling outputs 或一个 Value 的多个 data components 均不算多 publication event，不触发 PointEmissionMap。
- “有 exact parent”不等于“可当 FormalPulseScan y”。neutral-owned `FormalAssociationCapability` 是唯一判据：纯 one-to-one、固定配置的 selector/projection/processor 保留 upstream association并声明 ExactlyOne；固定 fan-out 必须在 bind 时声明 FixedPublicationEvents并携 PointEmissionMap；fan-in、跨事件 Fit、Histogram/Distribution 或任意无法逐事件重放的变换明确无 capability。frontend 只描述纯 transform facts，composition/neutral 根据 leaf 声明铸造 capability；TaskConsole 不递归猜。selector 配置改变会换 generation，正在 bind 的 scan 必须失败而不是沿用新 ROI。
- 不建立任何caller-visible software memory budget、byte quota、size estimator或预测性allocation rejection；内存不足由SDK/Python分配自然失败。formal/exact工作绝不因队列策略丢弃。Monitor Live Fit的one-active+latest只按“旧未开始revision已无产品价值”的latest语义整体supersede，既不按字节估算也不拒绝输入，不能演变成通用memory-budget机制。

### 5.1 Run admission 与硬件 lease

所有产品入口共享一个Experiment application owner：

```text
frozen RunPlan + owner/preemptibility descriptor
        |
        v
application admission method（一个内部owner，不是新framework）
        |
        +-- ResourceArbiter 在同一lock扫描全部exact claims
        |       +-- 无blocker -> 原子acquire all -> RunController
        |       +-- 有blocker -> immutable tuple（全部key与owner，不是首个）
        |
        +-- 全部blocker及其signal-dependent retirement closure
            均为同Experiment且effective-preemptible
                -> 同时请求正常retirement
                -> 等待每个cleanup/SAFE/lease release
                -> 对原frozen plan做一次最终admission
```

只要集合中有external/unknown/nonpreemptible owner，就一个也不自动停止；retirement后出现新racer则typed失败，不循环。用户在retirement中取消新请求时，已经请求退出的旧owners仍完成安全cleanup，但新Run绝不启动。Workbench、API、PulseGUI只观察`STARTING/RETIRING_CONFLICTS/RUNNING/FAILED`并显示完整冲突，不扫描UI rows、不重新prepare request，也不建立board-wide task lock。

`RunPlan.resource_claims`本身就是exact物理footprint，不再包一层无收益DTO：会configure/arm/fire设备的operation对真实adapter identity取exclusive claim；只读health/status走窄observation port；消费immutable signal的Fit/selector/Processor/analysis不claim上游device。每个leaf只声明自己实际驱动的exact设备。preemptibility属于frozen start owner而非ResourceClaim mode：continuous Camera monitor通常可抢占，finite acquisition与正式task默认不可抢占；外部/unknown owner永不可自动抢占。该application内部方法沿已经存在的exact signal dependency graph求retirement closure：若该monitor正在为FormalPulseScan/finite nonpreemptible consumer提供必需generation，它的effective preemptibility为false；若下游全是continuous preemptible processors，则它们与上游一起按正常lifecycle退休。不得让Processor伪claim上游device，也不得停上游后让正式下游静默失败。

当前产品的 exact claim 闭表是：

| operation | hardware claims |
|---|---|
| Camera Measurement live/finite | 用户选择的那一台 Camera |
| generic triggered Camera capture | 用户选择的 Camera + Sequencer |
| MOT Field | `mot_camera` + Sequencer |
| Pulse execution / FormalPulseScan | Sequencer；PulseScan 不 claim y producer 的 Camera/Processor |
| Release/Recapture、Duration Fidelity | 实际驱动的 Camera + Sequencer；只有计划真实驱动 RF 时再 claim 对应 RF |
| Fit、selector、普通 Processor、Calibration/Occupancy analysis与Figure | 无上游硬件 claim |

新增leaf必须按同一“实际configure/arm/fire哪个adapter”原则扩展该表和合同测试；不能用Task类别、signal依赖或UI所在窗口扩大claim，也不增加通用SHARED mode。

硬件lease的释放边界早于Run terminal：`hardware -> cleanup/DISARM/SAFE + device-buffer-sealed receipt -> revoke capability -> release lease exactly once -> artifact/report/finalize -> terminal state`。receipt必须同时证明SDK/DMA/ring buffer借用全部结束，post-safety输入已是进程自有immutable data；不能把会被下一次arm复用改写的view带过release。post-safety阶段无硬件能力；存储、报告或manifest inspection再慢也不能让安全空闲设备继续busy。

## 6. 运行、硬件、Frontend 与领域纵切合同

### 6.1 Task、Measurement、Processor 与 Fit

- Task 是一次用户 use case，可顺序组合少量 flat Runs；它不是递归 workflow、child plan 或全局调度器。
- Measurement 声明采集的物理语义和 output vocabulary；live/finite 只是同一 Measurement 的 host policy。Camera 只有一个 Measurement，具体使用 qCMOS 或 MOT Camera 由 typed device role/binding 决定。
- Processor 是 typed input 到 typed output 的领域变换。latest-only、exact、finite 是 delivery/execution policy，不产生新的 Processor catalog、form、binding 或 lifecycle。
- Fit 的数学值与 solver 属于 `zlc_data`；通用 Fit presentation 属于 `zlc_frontend`；运行宿主由 composition 注入。Fit 不是 neutral 通用 Processor，也不取得上游设备。
- selector 属于 Figure。Area 发布保留 dtype、axes、validity 和 exact selection provenance 的 Dataset；Cross 发布所选 sample/cell 的数据值，坐标只进入 provenance。Fit 只发布具名参数 sibling bundle，不发布拟合曲线数组；拟合曲线、点、中心或半径作为同一 frontend overlay 可见。
- 普通 pointer motion 不发布 hover 数据。Area/Cross/Fit 不重配 Measurement、不建立 ROI Measurement/Processor，也不弹出第二个 DataFigure。

每个 Logic Node leaf 关闭自己的 Definition、Request/Config、专属算法、输入输出合同、prepare/evaluate/materialize、resource claims 与 artifact lineage。普通字段、choice、dynamic output 和 default FigureIntent 全由 inert declaration 提供；Workbench 不按 DefinitionKey 写具名分支。

### 6.2 Flat Run、线程、取消、cleanup 与资源

`RunController` 执行同步、扁平的 `RunPlan`。设备 session 的 owner thread 是唯一 configure/arm/read/stop/close 调用者；Qt、API 和领域算法只持窄 command/result port。Cancellation 是 cooperative typed state，不用 kill thread、共享 bool 或 GUI row 删除伪造 terminal。

Run 生命周期固定为：

```text
prepare frozen plan
-> unified admission
-> acquire exact claims
-> hardware execution
-> cleanup / DISARM / SAFE / device-buffer-sealed receipt
-> revoke drive capability
-> release hardware lease exactly once
-> validation / artifact staging / report
-> terminal publication
```

terminal 只能在 worker、session、interrupt 和 cleanup 全部真实退出后发布。cleanup failure 使本 Run 失败并保留 primary/cleanup diagnostics，但不制造跨连接 quarantine 或持久门禁。新连接只凭实时 identity、当前 SAFE 初始化和当前 capability 建立 authority。

有限采集按冻结的 expected cardinality 完整配置设备 buffer；continuous monitor 的 rolling history 是 Dataset/product cardinality，不是内存预算。formal/exact 事件不得因软件队列策略丢弃；monitor 可以按明确 latest 语义整体 supersede 尚未开始且已经没有产品价值的 candidate。

### 6.3 Pulse、FPGA、Camera 与 FormalPulseScan

`zlc_pulse` 独占 PulseDocument、target manifest、compiler、deployed geometry、typed execution observation、transport/server 和冻结硬件资产。neutral sequencer application只消费其 public values/ports，不能解析 transport 私有字典或复制 geometry loader。

`zlc_pulse` 的 public observation 是 strict-codec、immutable typed value，不是 `dict[str, object]`：

```text
PulseDeploymentObservation:
  connection_generation
  target_manifest_digest
  deployed_geometry/build_fingerprint
  session_state + prepared_artifact_digest | None

PulseExecutionObservation:
  connection_generation + execution_id
  artifact/program/schedule digest
  execution_form + state
  expected/completed trigger counts
  typed progress/fatal/underflow/terminal facts

PulseSafeReceipt:
  connection_generation + deployed identity
  exact current readback proving existing outputs are SAFE
```

raw backend maps只可附在diagnostics，不能参与neutral分支、artifact admission或SAFE铸造。只有拥有真实session/readback的`DeployedStreamerSession`能在同一connection generation内签发SAFE receipt；retry在同一owner lock串行，失败拒绝本次连接但不建立quarantine。上述values只包装现有status/register/transport事实，不要求RTL、ROM或bitstream增加字段。

Camera与running-signal association使用两种不可互换的既有边界，不建立receipt类型家族：

- physical finite capture复用现有`CaptureStartedAck`：它证明旧acquisition已经stop/drain、new arm建立、source ordinal baseline为零、预期cardinality和buffer配置已冻结；session/binding identity提供single-use边界。
- FormalPulseScan over an already-running signal复用现有association cursor的私有token：它冻结producer generation、artifact/schedule fingerprint、publication cardinality、stable counter/event baseline、operation deadline与当前working-point quiet fact；generic association port不持Camera binding、不重arm或重配上游。
- 两者都绑定exact request/adapter identity、不可跨generation复用；FIRE后的terminal/count/stamp reconciliation必须匹配该边界。token缺失、重复消费、identity变化或deadline越界都在artifact commit前使Run失败。

正常 Pulse 与 SCAN_SLOT/MOT 执行只使用现有 bitstream 的 autonomous streamed hardware timing。API-slot 无法无缝更新时才允许既有、显式标记的 segmented `STATIC_ONCE` 路径。不存在逐 cell host fire/wait、software sleep timing 或为了架构偏好新增 trigger FIFO/counter/ROM attestation 的 baseline。

FormalPulseScan 只拥有：

```text
frozen pulse program
+ sequencer Run
+ 已经运行的外部 Signal(y)
+ producer-owned FormalAssociationCapability
+ exact terminal/coverage evidence
```

它不取得 Camera、Processor 或 Figure 的设备，不启动/停止上游，也不按 producer 类型建立第二条 capture pipeline。普通 cursor 只证明软件交付顺序；正式 scan 必须在 FIRE 前冻结 publication cardinality/association，在 FIRE 后绑定 exact pulse terminal，并在 commit 前验证每个 event、coverage、generation 和 lineage。

qCMOS 正式资格使用现有硬件能力，并明确区分“路径语义资格”与“本 Run 事实”：

1. 一次 arm 为外触发模式，并按整 Run 的冻结帧数配置；
2. E0 qualification 只主动证明当前 adapter/连接、ROI、binning、readout、trigger wiring/mode 与 counter/stamp 的 ordered one-frame-per-trigger 语义；不得由少量固定 trigger 虚构最大 scan count、最大 delivery latency或无限 sustained-delivery envelope；
3. 每个 Run 以 pulse artifact 为完整 schedule 的唯一 owner；association 只冻结常数级的 artifact/schedule fingerprint、channel、count、minimum spacing 与 clock，不能把 O(N) schedule arrays 再写入 lineage；
4. preflight 读取当下冻结 exposure/工作点的真实 minimum trigger interval并核对完整 compiled schedule 的最小间距；不满足则 FIRE 前拒绝。新 finite Capture 在 physical arm 时把设备 buffer 配成该 Run 的 exact cardinality。已经运行的 signal association 绝不重arm、resize或重配上游，而是在 FIRE 前冻结当前 arm/session 与 stable produced/drained/publication baseline，依靠上游持续排空有限 driver ring 和无损 raw stream/FollowTap 交付；ring overrun、少/多/乱序、stream publication failure或期限届满都使整 Run INVALID。它保证 fail-closed 正确性，不承诺任意长 schedule 在固定 ring 上必然成功。只改变 exposure 时，endpoint须证明其它 qualification-scope facts未变并用新 readback重新计算 timing/quiet facts；
5. Run 末端比较 expected/emitted/produced/observed/drained count，检查 frame/camera stamp 与 timestamp 的单调性，再经过由当前物理 trigger interval 派生的 quiet window确认counter不再增加；普通SDK timeout只作为本次失败期限，不能命名或持久化为“qualified maximum delivery latency”；
6. 任一缺帧、多帧、乱序、late extra、counter倒退/wrap歧义、generation change 或 terminal/coverage 不一致使整 Run INVALID，不能提交、不能自动重跑。

该保证是 preflight + per-run reconcile，不声称具备现有硬件没有的逐沿 tag。只有 E0 或代码证据证明现有 RTL 真 bug/偏离既定设计时，才单独评估与根因直接相关的最小硬件变更；任何硬件修改都不由本文自动授权。

### 6.4 Frontend、Figure、render 与表单

`zlc_frontend` 是全部通用 Figure 的唯一 owner：

```text
FigureIntent
-> ViewContract/default_view
-> PlotPanelContract/Session
-> immutable FigureFront
-> shared Qt host or headless export
```

TaskConsole、Calibration、Occupancy、DataFigure、FigureViewer 与 Pulse preview 只构造领域 intent 或消费同一 contract，不手写 composer、Divider、style、panel size、DPR、default axis、selector geometry、Fit overlay 或 export codec。正式用户 plot kinds 只有 `2d/sites/1d/monitor/hist/grid`；静态数值 primitive 不是可添加 panel。

公开 Figure vocabulary 只有一个 typed 闭集：

```text
PlotKind = IMAGE | SITE_MAP | CURVE | ROLLING |
           HISTOGRAM | GRID | PULSE | METER
```

`2d/sites/1d/monitor/hist/grid`只是上述前六个可添加产品面的稳定UI key/label映射；PULSE和METER只服务专用/内部surface。Distribution是HISTOGRAM的display-only双高斯/threshold分析，不是第二个PlotKind。所有public Figure contract、archive、renderer dispatch与Add Panel菜单都消费该typed value，不能各自保存string vocabulary。

GRID是一个带恰好一个typed `FACET` binding的容器，不是renderer family。它必须保存`cell_intent ∈ {CURVE,HISTOGRAM,IMAGE}`及该cell intent的完整View/Display contract；facet source可为R、trailing data axis、PointRows、PointCoordinate或GridDimension。每个focused cell复用普通PlotPanelSession、selector、Fit与export，稀疏hole保留typed address并禁用不适用动作，不能暗选第一格或按shape猜cell kind。

Saved Fit Grid是保留的结果浏览产品，但其navigation model只拥有typed address/selection/previous/next/label和构造时建立的ordered occupied-address index；page navigation为O(1)或O(log N)。所有raster、form、style、selector与Refit走同一`DataFigureRenderSession -> PlotPanelSession`，不得拥有专用render session、repeat/facet preferences或第二composer。

`FigureFront` 原子携带 base raster、exact source/projection、完整 display state、RenderedGeometry、selection 与 typed overlays。Divider/data box、coordinate mapper、font/color/style、logical size 与 DPR 都来自同一 frontend owner。Qt 不先 stretch 旧 raster 再等待新 raster；resize/DPR 改变使旧 geometry 失效并触发同 revision recompose。交互中的轻量 overlay 使用 painter/blit；pointer drag、Fit completion 和 selector update 不重画 Matplotlib base。

Frontend 同时拥有每个 plot intent 的唯一 canonical display form。Setting、Edit、DataFigure、FigureViewer、Calibration 和 Grid cell都消费相同 `PlotDisplayFormSpec`、handler 和 authored state；Workbench只放置 editor、绑定 panel 与提交 Apply/Cancel。

Matplotlib的canonical字体与main产品视觉合同一致：由`zlc_frontend`/`[render]`打包并注册仓库内唯一的`Helvetica Light` TTF；Qt Windows UI固定使用共享token中的Segoe UI。字体文件必须作为正式产品资产跟随wheel/sdist并在启动时校验family；不允许leaf/report用局部font family、glyph workaround或第二style表改变同一FigureIntent的像素。

控件投影规则：

- 严格正向 `bool` → `FluentSwitch`；
- 固定 2–3 个短、对称且不会扩展的 enum → frontend segmented choice；
- 需要同时展示解释的少量 mode → radio/card group；
- 动态、可扩展、长标签或超过三项 → `FluentComboBox`；
- numeric widget 只承载 numeric；AUTO/Select/Inherited/Unavailable 是外部 resolver/null state，不得编码成字符串或特殊数值 sentinel；
- conditional enable/visibility 由同一 frontend form presenter派生，不能由 Setting/Edit 各写一份。

普通 Qt draft 原位 reconcile 稳定 widgets。Add/Remove/Reorder 只增删移动对应子树；unit/name/value/delay/binding/visibility 不重建整树。只有 document generation 或 target topology 真替换才允许全量 replacement。

已经进入正式 Qt object tree 的 QWidget 从构造起由最终 host/pane 拥有。不得用 `setParent(None)`、parentless candidate 或临时 top-level 完成复用、排序或异步 admission；重排直接在稳定 container 中移动，终态删除使用 `hide() + deleteLater()`，完整 subtree 构造并验证后一次挂载。

新 Logic-node row 必须先由 owner declaration 的 defaults 形成完整、领域有效的 typed draft，之后才可进入 signal topology、dynamic output 推导或 request 构造；空白或部分构造的 GUI placeholder 永远只是 presentation state，不能成为领域输入。

Logic tree与signal picker只有一个generic projector，只读显示当前typed publication的shape、dtype、unit、generation与formal-association状态；history/buffer属于运行元数据，永不进入shape。尚未发布时显示`—`，已有value却缺DatasetSchema时显示typed contract error；任何leaf不得创建专用dimension/signal row或可编辑shape字段。

GUI 证据有两条正式路径：快轨使用 offscreen + `ensure_qt_app()` + 正式 composition + 真实 Qt input + outer grab；慢轨从正式 launcher 按人类流程运行。手工假 QWidget、直接调 controller、无文字截图或另一套 sizing/style 不能验收。

### 6.5 Live Fit、Snapshot Fit 与性能边界

Monitor/Setting Fit 是 surface-scoped live session，绝不隐式 Hold/Pause。base 持续呈现最新 front；每个 session 至多一个 active candidate 与一个 latest-pending candidate，跨 session 公平。结果保留 exact source/projection provenance，overlay 显示 CURRENT/LAGGING 或在不兼容时隐藏。Fit 参数作为 `EVENT_RESULT` 发布，不进入 continuous causal frontier。

Edit/DataFigure Fit 是 snapshot-scoped invocation，只修改该 snapshot surface 的 overlay/result；它不读写 Monitor Fit state，不冻结任何 Monitor panel。显式 Pause、pointer gesture 与 Edit snapshot可以持有自己的 base，Fit 按钮本身不能。

Fit compute、base raster compose 和 selector Dataset materialization是三类独立 operation。Qt 只 O(1) submit immutable ref/spec并接收小结果。规则 H×W 图像使用 data-owned regular-raster problem：保留原 dtype readonly view，坐标用 O(H+W) axis vectors，矩形 ROI 用 slice/view，validity 用紧凑 typed mask，seed/moment 用浮点 accumulator，score/solve 分 chunk或只处理选定样本；不得生成两张 H×W coordinate grid、多套 full-size float64 copy或 Python逐元素检查。执行后端只在复杂度修正后的 profiling 证明需要时选择 thread/process。

Fit源码按真实依赖边界拆分，而不是按UI入口复制：`zlc_data`分别拥有contract、closed model catalog、problem packing、solver、codec与窄public facade；problem是唯一packing owner，solver不导入frontend。frontend的Curve/Image/Histogram projection因输入数学与几何不同可分别成模块，但共用一个batch/source mapping与同一authority translator；安全、不可执行表达式的Fit argument parser保持独立。不得把这些职责合成kind-switch巨模块，也不得为TaskConsole、Calibration、DataFigure或FigureViewer再复制一套fit文件。

Distribution 的双高斯与 threshold 是 renderer 对冻结 histogram bins/counts 运行的窄 display-only analysis；它使用同一 model math/style，不发布 Fit 参数、不修改 authored histogram state。显式 Figure Fit 存在时覆盖该显示分析。

### 6.6 固定 namespace、leaf UI、public API 与产品流程

内建 device 与 Logic Node 使用固定 namespace、冻结 descriptor 和确定性 discovery；不是 plugin registry、entry point、service locator 或中央 concrete import 表。Experiment 在对用户可用前完成 discovery、dependency graph、installation binding 与 typed unavailable validation，并把 API/Workbench 所需 facts 冻结。

ordinary leaf 不含专用 TaskConsole form/presenter/binding。只有 declaration + generic Figure 无法表达的真实产品交互才允许 inert `ui/**`，它只导出 lazy `UiContributionDescriptor`，依赖 frontend generic UI context与窄 command/load/save ports；leaf UI不得导入 Workbench。headless discovery不加载Qt，产品启动时解析失败必须明确失败，不能静默退回第二 UI。

baseline允许的optional leaf UI闭集只有：PulseScan scan-table/slot editor、Calibration creation/multi-page report surface、Occupancy exact-cell navigator。它们的普通fields仍由declaration projector生成，Figure/SiteMap/selector/style仍委托frontend。新增例外必须先用真实产品交互证明generic declaration + Figure无法表达，并在修改本架构后才可实现；“布局方便”“字段较多”或leaf想自定义风格都不构成理由。

`Zou_lab_control` 是脚本、notebook 与 desktop 共用的唯一 public application API。它暴露稳定 `Experiment`、`exp.nodes.*`、device/application lifecycle和Workbench opener，只做窄委托；不拥有领域 schema、算法、materializer、Figure 或第二运行时。DeviceManager 是 desktop 的默认 Experiment composition入口；TaskConsole、PulseGUI 与其它窗口共享同一个 Experiment application owner。

关键产品纵切必须保持：

- Camera Measurement：同一领域节点可选择 qCMOS 或 MOT Camera；live signal固定 `(1,1,*frame_shape)`，finite N帧从progress到FINAL固定 `(N,1,*frame_shape)`并用validity表示未完成；history不进入signal shape，源 dtype（通常uint8）不被显示改写；同一cycle多帧是一个atomic sibling publication。
- Calibration：内建 readout能力，不是plugin。capture与calibrate是两个linked flat Runs；命令宿主不是第三Run。结果只通过同一 frontend FigureIntent/report contract呈现，SiteMap物理事实与CalibrationArtifact属于neutral owner。
- Occupancy：必须显式接受已有CalibrationArtifactRef；current calibration只作为可见可改的默认ref注入。它消费同一 shot 的frame/calibration facts并原子发布typed siblings，不由Workbench拼接。
- MOT Field：Ready、Running、multiple live updates、FINAL artifact/default grid view全部可观察；point rows保持authored顺序，标量输出为 `(R,P,1)`，GridTopology只描述真实grid；accumulator不能随采样数平方复制。
- PulseScan：只消费已经运行且具正式association capability的signal；repeat sweep形成R，pulse内部RepeatRegion不改变R；P来自冻结point table，非grid轨迹保持一串authored rows。
- Figure-derived Area/Cross与Processor连续输出进入同一 explicit parent transaction graph；Fit参数是事件结果。ROI→ROI→Fit、三份以上仍被operation借用的revision和independent producer都不能依赖name/latest重建因果。

每个领域切片必须从正式 Experiment/SignalPlane/PlotPanel入口验证，不以单元算法、synthetic renderer或旧测试适配替代产品流。

## 7. Artifact commit 的唯一终态：manifest-only

用户可变路径由composition创建一个immutable `WorkspacePaths(pulses_root, tasks_root, output_root, repository_root)`，并把所需的窄Path传给leaf/Workbench。`zlc_storage`只提供`resolve_under(root, path)`、canonical bytes、CAS与atomic I/O，不从package位置、CWD或环境猜project root。deployed geometry、target manifest、RTL/bitstream等immutable packaged/deployment assets由各自package resource或显式deployment config拥有，绝不塞进WorkspacePaths。

Artifact commit 固定采用 manifest-only 协议。CAS manifest 路径由 canonical payload digest 决定，immutable manifest 的 atomic publish 是 artifact 唯一可见性与线性化点；load/admit 不依赖第二份持久 marker、intent 或 recovery journal。

```text
hardware terminal -> cleanup/SAFE/join -> release hardware ResourceLease exactly once
        |
        v
验证全部物理/cardinality/terminal/lineage facts
        |
        v
stage + fsync immutable blobs（仍可 cancel；留下 blob 只是安全 orphan）
        |
        v
mint process-local single-use PreparedArtifactCommit
        |
        v
RunController 最后检查 cancel/deadline，原子关闭 cancel gate
        |
        v
atomic publish exact canonical manifest + durability barrier
        |
        +-- success --------------------------------> return typed ref
        |
        +-- publish exception -> inspect exact expected target only
                                  |
                                  +-- exact bytes+digest visible -> success，同一 ref
                                  +-- confirmed FileNotFound     -> original failure
                                  +-- wrong bytes/digest          -> repository corruption
                                  +-- storage temporarily unreadable
                                      -> COMMIT_INSPECTION_PENDING
                                         只重试 read/confirm，不重试 publish/FIRE
```

闭合规则：

1. `PreparedArtifactCommit` 是不可序列化、一次性、进程内 capability；它只封装 run_id、expected typed ref、canonical manifest bytes、publish-once/inspect callbacks 与 repository borrow。RunController 在一个 lock/state transition 中消费它，防 cancel/commit race；cancel 先赢则 publish 次数为 0，commit 先赢则 cancel 不可反悔。不需要 commit_id、intent 或 durable Run state。
2. hardware ResourceLease 在 terminal cleanup/SAFE/join receipt 后、artifact validation/staging 前恰好释放一次；cleanup error 先使 Run 失败，不得进入 commit。唯一 owner 明确为 `RunController` owner thread；它持有 `PendingManifestInspection`，RunSnapshot phase 为 `commit-inspection-pending`，该 Run 此时非 terminal，`result()/wait()` 可超时。commit gate 已关闭后 cancel 只返回 too-late，绝不把状态改成 CANCELLED。既有 owner loop/poll/wait 只执行同一 target 的 read/confirm且同一时刻最多一次。repository borrow 只防 close，不是 write authority、不阻塞新 commit；后续成功/失败不得再次释放硬件。
3. application shutdown 对每个 pending operation 做一次 bounded final inspection；仍不可判断则放弃 process-local handle并释放 borrow，返回明确 indeterminate shutdown diagnostic。它不 republish，manifest 仍是唯一真相，startup 没有 pending gate。旧 Run pending 或 handle 被放弃时，新 Run 均可使用同一硬件/仓库。
4. 进程崩溃后没有旧 caller 要恢复：已持有/持久化 known typed ref 的 caller 可直接 load/admit/has exact manifest；manifest 不存在时只有 blob orphan。若 ref 未在别处持久化，即使 manifest 存在也视为不可发现 orphan；baseline 不承诺 repository enumeration、最近 Run 恢复或自动补写 pointer，亦不为此增加 journal/第二 authority。
5. known-ref `load/admit/has` 只验证 typed ref、canonical manifest digest/bytes、referenced content digests、artifact schema、terminal/lineage/provenance；不得查询持久 side marker。Capture、Scan、Calibration 与 Occupancy 的 run_id、source refs 和完整 provenance 都进入自己的 canonical manifest，并在 stage/load/admit 由该 artifact owner 验证。
6. `RepositoryRootLease` 只保证单 writer，ContentStore 只负责 immutable CAS、atomic replace 和 durability。产品路径不存在持久 commit journal/coordinator/intent、第二 visibility marker、startup recovery gate或通用 framed side log；没有其它消费者的对应源码、exports 与 tests 同时删除。
7. qCMOS/terminal/cardinality/lineage INVALID 必须发生在 mint/publish 前；manifest-only 协议不会把确定性合同错误 recover 成成功。lost ack 仅由 exact bytes inspection 认定成功。

## 8. 依赖闭合的实现顺序

不得逐文件打补丁并长期保留双模型。每个 M1–M6 cut 替换生产 owner、public contract 或 codec 时，直接依赖旧语义的测试源码必须在同一 cut 改写为当前物理/public contract，或与被删行为一起删除；M7 只执行 broad suite、补跨 cut E2E 和做最终总清点，不接收此前遗留的旧测试。顺序固定如下：

### M0：规范冻结

- 建立唯一 System Architecture；AGENTS只保留执行协议，Maintainer只保留checkpoint，清除重复或冲突的设计文档。
- 先冻结不可变量、owner、产品流和允许的最小public concept vocabulary。PointTable/GridTopology、source binding、fit authority、transaction/front/association、typed pulse observation与PlotKind/Grid contract等架构名词不自动对应一类一个文件；优先复用现有value/function并原位替换旧模型。
- 本步不为了旧测试保留任何 alias。

### M1：Point domain 最小替换

- 先在现有data owner内原位替换point truth：P row ordinal是identity，coordinate是相关column，GridTopology只作可选metadata；不得在旧`point_axes/PointLayout`旁新增第二套resolver/descriptor/codec。
- 在同一未完成cut内先用一个真实非grid producer→Dataset→Figure/Fit和一个真实grid producer证明同一模型；核心owner闭合后，其它producer/consumer只做机械call-site迁移，不再各自增加source wrapper。
- 最后一个生产reader迁走时，同一cut删除被替代的Dataset point writer/reader；不得为了历史/phase tests保留compatibility。直接依赖被替代模型的测试也必须在同一cut按当前物理/public contract重写，或与旧行为一起删除；不能把旧测试源码拖到M7。上述纵切只是实现顺序，不是可提交的兼容阶段；Git checkpoint/commit与产品验收点均不得有新旧public模型并存，也不得增加migration adapter。

### M2：Signal transaction

- publication冻结一个immutable sibling bundle、exact parent refs与单调sequence；一个SignalPlane owner维护generation/route/frontier，禁止resolver/plane/publisher/Window各存一份replacement状态。
- one-to-one、fixed fan-out与无formal association是同一owner的三种明确结果；Emission slot/map只在真实fixed-event PulseScan消费者需要时出现，不因文档名词预建通用framework。
- neutral generation只冻结自身route/schema/cadence；UI attachment进入topology前一次验证并冻结对应frontend静态FigureIntent。每revision只有neutral exact SignalFront，Workbench从它与attachment intent生成并原子呈现完整board front。presentation部分不新增public type；prepare/candidate/completion只是owner私有局部状态，不建立Presented/Prepared/Admission/Route DTO。
- Area/Cross continuous与Fit EVENT_RESULT是独立routes；SiteMap物理事实来自exact signals/artifact而非sidecar。删除PresentedSignalPublication/PresentedSignalFront、ConsolePresentationIndex、动态混合Area/Cross/Fit的FigureOutputSession、tick内topology修复、latest/name parent重建与Window第二generation owner。先以raw→ROI双panel、ROI→ROI→Fit、至少三retained revisions、independent producer和FormalPulseScan五条产品流证明模型，再机械迁移其它调用点。

### M3：Figure/View/Fit 收敛

- 删除legacy repeat enum，建立直接SourceViewBinding、§4可执行default policy、EvaluatedProjectionFront+RenderedGeometry与唯一selector engine。
- 建立single-authority CommittedTransform/FitSpec translator；Fit source统一为相对effective schema解释的AxisSourceRef，Fit结果publication与overlay currentness分离。
- 删除grid-specific renderer/preferences、frontend FigureFitLane、DataFigure第二executor与card-global Fit runtime；只保留composition-owned窄compute submit/cancel/completion seam及现有serial render owner。Monitor live/latest与Edit snapshot按surface私有状态分离，删除Fit隐式Hold。regular raster dispatch留在现有fit_problem/fit_solver内部，不新增public problem/lane/session类；overlay painter/blit不重画base。
- 收敛typed PlotKind、字体、public barrel与Calibration/FigureViewer共用的frontend FigureIntent入口；每kind只留一个已有frontend canonical display FormSpec，不新增包装DTO，删除Workbench第二套relim/limits。bool/choice/nullable numeric按§6.4统一投影，leaf ui反向Workbench imports同切片清除。

### M4：Run、pulse、storage 边界

- 收敛SAFE唯一owner/receipt、typed progress、single geometry loader、neutral→pulse public API；保持RTL/bitstream冻结。
- 在现有finite-start ack与association token内分开physical arm和running-signal association；按§6.3完成路径语义E0、完整schedule的常数级绑定、当前工作点preflight与per-run quiet-window reconciliation，不新增receipt类型家族。
- 将hardware lease释放移到SAFE+device-buffer-sealed receipt/capability revoke之后、任何artifact/report之前；以success/failure/cancel覆盖exact-once release，post-safety操作不得持SDK/DMA view或延长busy。
- 删除commit journal/coordinator/intent/reconcile与无消费者generic framed side journal；实现PreparedArtifactCommit、manifest atomic publish、PendingManifestInspection与success/lost-ack/absent/unreadable/corrupt矩阵。Calibration/Occupancy provenance进canonical manifest；引入composition-owned WorkspacePaths。

### M5：叶包、composition 与公共 API

- Experiment在可用前发现并冻结leaf/device descriptors，验证全依赖图，同时把API与TaskConsole binders收窄为declared frozen facts；删除中央pulse catalog/fake DeviceCatalog dynamic fields。
- 在现有Experiment application owner中实现唯一内部admission方法；ResourceArbiter一次返回完整blocker immutable tuple，application沿exact signal graph求retirement closure，按effective preemptibility全退或全不退，再对原plan一次最终admission。CommandContext只需窄`start_and_wait(frozen_starter)`能力，不建立Admission/Conflict/Policy类链。删除TaskConsole conflict scanner与board-wide task lock；public facade只委托narrow ports/Workbench opener，release-recapture helper只留family内。
- optional leaf UI用lazy UiContributionDescriptor与frontend generic context；新增纯Logic Node只改leaf，无中央concrete switch。

### M6：领域 vertical slices 与性能闭环

- Camera一次性收敛live/finite shape、N-frame atomic siblings、role discovery、selection-aware freeze；PulseScan修正R/RepeatRegion、删除scan_shape并只消费association port；Capture/Readout按M1/M2 identity验收。
- Calibration删除第三Run和pooled pseudo Dataset，按两linked flat Runs、post-FINAL warning、single SiteMap FigureIntent闭环；Occupancy explicit calibration ref/provenance闭环。
- MOT修accumulator O(N²)并跑通Ready→Running→multiple live→FINAL/default grid view；release-recapture及其它leaves逐个用同一generic lifecycle验收。
- 每个slice都走真实Experiment/SignalPlane/PlotPanel产品入口，不以单元算法或synthetic renderer fixture代替功能验收。

### M7：证据与清理

- 完成 point/source-binding/group property tests、same-shot SignalFront/transaction tests、fixed-cardinality/association tests、live/snapshot Fit隔离与nested ROI replay、regular-raster profile、calibration canonical-raster parity、complete-blocker admission/lease-release、semantic widget projection、SAFE retry、manifest pending/lost-ack matrix、Camera路径语义E0+实际Run长schedule对账合同和正式product E2E。
- 对M1–M6每个slice记录旧树等价能力生产行数比和抽象consumer/invariant清单；>约3倍的slice必须在合并前完成压缩或给出逐项物理/边界理由，不能用测试/codec/历史兼容凑解释。
- 总清点此前各cut已经同步改写/删除的测试，补齐跨cut property/product E2E；不得在M7才首次处理旧owner测试。清除过期tutorial、重复架构文档和所有死symbol；保留与main视觉合同一致且由唯一style owner注册的Helvetica Light正式资产。
- 最后才跑 broad suite；这里延期的是整套执行，不是测试源码迁移。失败按仍有效的物理/public contract 判断，绝不恢复旧架构迎合历史测试。

## 9. 最终验收门

只有同时满足以下条件才可声称迁移完成：

1. 生产代码、public surface、codec、文档与current tests中不存在任何被替代的数据布局、repeat enum、专用grid/Fit renderer、frontend FigureFitLane、Fit隐式live pin、PresentedSignalPublication/PresentedSignalFront、per-revision presentation sidecar、ConsolePresentationIndex、动态混合Area/Cross/Fit的FigureOutputSession、GUI冲突裁决、重复display form、numeric文本sentinel、pseudo task、retrospective transaction、持久commit side authority、中央concrete catalog或fake catalog state；死symbol、alias、reader、wrapper与目录均为零。
2. regular/sparse/serpentine/arbitrary/duplicate/P=1 PointTable 全部 round-trip；canonical scalar/role 闭集拒绝非法值。`resolve_point_rows` 的普通 source tuple 与 `ResolvedPointRows` 对 row/coordinate/topology order、membership、missing count及FitResultBatch descriptor精确round-trip。source-role组合枚举全部fail/pass正确，POINT_ROWS/POINT_ORDINAL/point_ordinals/SELECTED无双authority；一般非网格多维scan、多POINT_COORDINATE Fit/group、P×data-axis Image均不Cartesian-expand，每ordinal只有一个observation address，只有GridTopology路径densify。
3. PulseScan 改 RepeatRegion 不改变 Dataset R；改 sweep count 只改变 R；硬件仍一次 autonomous stream。
4. selector/Fit/Processor都携exact parent；neutral SignalFront保证raw+derived connected panels只原子提升完整frontier，siblings不拆、无关producer可独立前进。每个UI attachment的generation-static FigureIntent在进入UI topology前一次冻结，每revision无presentation sidecar；Workbench只能由同一个exact SignalFront生成全部FigureFront并原子present完整board，attachment失败不阻止neutral signal。one-to-one/fixed-fanout/fan-in的FormalAssociationCapability判定唯一且PulseScan不靠heuristic；FixedPublicationEvents扩展有grid的base table时slot dimension保持row_to_cell injective。
5. Calibration、TaskConsole、DataFigure、FigureViewer对同一FigureIntent/source/complete display state/size/DPR使用同一canonical PlotPanel base raster与frontend overlay contract；FigureFront含actual bins/group/effective schema及与base raster同compose的RenderedGeometry。source front swap期间gesture锁旧front，resize/DPR invalidation则无final publication地cancel并recompose，绝不跨geometry映射。Cross/Area连续交互同源且无hover；Live Fit期间base持续前进、Edit Fit不影响任一Monitor，overlay携独立source ref并显示CURRENT/LAGGING或隐藏INCOMPATIBLE，Fit params仅EVENT_RESULT。standalone不伪造signal，ROI→ROI→Fit及三个以上retained revisions无presentation/transaction race。
6. Camera live shape固定`(1,1,*frame)`，finite从progress到FINAL固定`(K,1,*frame)`并以validity表示未完成R；history不影响shape且dtype保持`uint8`。N=3的frame_0/1/2同transaction。MOT Ready→Running/live/FINAL，shape`(R,343,1)`、topology`7×7×7`且不O(N²)。Calibration恰有两linked flat RunId、command host无RunId；Occupancy可显式加载已有CalibrationArtifactRef且current ref只预填。Camera(`camera`, live或finite)+MOT并行；Camera(`mot_camera`, live)+MOT只退休该monitor；Camera(`mot_camera`, finite/formal)明确拒绝且不先停。多资源冲突一次返回完整集合、全退后对同一plan只admit一次；nonpreemptible/外部owner不被自动停止。若camera monitor正供FormalPulseScan使用则不被抢占；纯continuous downstream则随retirement closure正常退出。cancel/post-FINAL warning正确。
7. 冷启动/崩溃重启均在任何admission前完成geometry handshake+由DeployedStreamerSession基于真实readback铸造的SAFE receipt；故障可重试、并发retry只有一个物理SAFE、无quarantine。artifact lost-ack exact inspection返回同一ref；pending Run非terminal、cancel too-late、wait可超时，hardware lease已释放且新Run可继续。Calibration/Occupancy known-ref manifest都可直接验证run/source provenance，无journal。
8. 真机E0主动证明当前adapter/连接、固定结构工作点与counter/stamp的ordered one-frame-per-trigger语义，但不由短测试虚构最大count或delivery latency。finite Capture走physical arm/zero baseline并配置exact buffer；PulseScan over running signal走独立association token，冻结已有arm/session与produced/drained/published baseline，不重arm或resize。每个Run对pulse-owned完整schedule做常数级fingerprint/channel/count/min-spacing绑定，以当前工作点readback做preflight；FIRE期间上游持续排空有限driver ring并无损发布，FIRE后验typed terminal、produced/observed/drained/published与stamp order，再经过当前物理quiet window确认counter持续exact。任何ring overrun、stream publication failure、late extra、少/多/乱序/wrap歧义在PreparedArtifactCommit前使整run INVALID，不自动重跑；该模式保证fail-closed，不保证任意长Run都成功。
9. installation graph对API及TaskConsole requirements的missing/ambiguous/cardinality/cycle启动即失败；两个binder只收frozen narrow facts且不保留catalog。新增/删除纯Logic Node只改叶包；新增物理device才改device leaf/部署配置；无中央concrete switch。唯一系统设计文档、代码、public API、教程和当前合同测试完全一致，无compatibility residue。
10. M1–M6每个slice都有main等价能力生产行数比与抽象consumer/invariant清单；>约3倍均已压缩或逐项证明物理/边界必要性。不存在只为单一成员/单实例/旧兼容而保留的enum、wrapper、DTO或目录。
11. 正式Qt快轨确认unit/name/value/selector program编辑不创建全局snapshot或重建无关widgets，Add/Remove/Reorder只修改对应结构；render/submit/hardware/publication边界才冻结。交互性能问题先以profile证明compose/copy/lock根因，再优化唯一owner，不加防抖或假缩放掩盖。
12. 2304² uint8 Live Fit不改变source dtype、不生成H×W坐标grids或多份float64全图；Qt heartbeat/wheel/drag/resize无Fit造成的>100ms stall，GUI callback p95<50ms，Fit完成不触发base Matplotlib compose。两个panel同时Fit公平推进；执行后端选择有修正后profile证据而非预设process。普通Hist和Grid-Hist的Log count均为同一bool FluentSwitch；Setting/Edit/Viewer/Calibration共享一个display FormSpec；生产numeric widget无AUTO/Select special-value sentinel，dynamic enum仍是Combo。

## 10. 收敛结论

最终架构不需要dense multidimensional ndarray、异步工作流编排器、硬件重构、persistent safety quarantine、软件内存预算、per-revision Presented*、通用borrow/lease framework、为Fit预设的独立进程、三套operation lane或第二套Figure/form renderer。真正必须做的是：以PointTable修复point identity；以ProducerTransaction/SignalFront修复same-shot，以UI-attachment-frozen static intent+exact SignalFront修复presentation lineage；以canonical ViewSpec/explicit FitSpec和surface-local live/snapshot语义修复显示与权威边界；以现有solver内部regular-raster dispatch修复大图Fit复杂度；以一个application admission owner+post-SAFE lease release修复误reject；并把SAFE、progress、geometry、storage path、form projection和leaf binding放回唯一owner。

M0–M7 是上述 owner 与合同的依赖闭合顺序，不是可并存的阶段架构。每个切片只允许一个终态实现；最终资格只由 §9 的完整证据决定。
