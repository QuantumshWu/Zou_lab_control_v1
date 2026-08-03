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
  zlc_data       : Dataset/Point/Validity/immutable value facts
  zlc_storage    : path confinement/atomic files/small-metadata encoding
  zlc_pulse      : pulse document/compiler/deployed transport facts -> frozen FPGA/RTL

zlc_plot          -> zlc_data
  PlotSpec/projection/selector/Fit/Matplotlib/raster/style/live presentation

zlc_frontend      -> zlc_plot
  shared Fluent widgets/typed forms/window shell/Fluent control projection

zlc_neutral_atom  -> zlc_data + zlc_storage + zlc_pulse + zlc_plot.kinds/specs
  devices/Logic-Node leaves/Run/SignalPlane/artifact lineage
  optional leaf ui submodule -> zlc_frontend + zlc_plot（inert discovery不加载它）

zlc_workbench     -> zlc_frontend + zlc_plot + zlc_neutral_atom
                     （必要时消费zlc_pulse public facts）
  Qt product composition only

Zou_lab_control   -> 上述 public contracts
  public Experiment API + composition + application lifetime
```

禁止的反向边包括 data/plot/frontend→neutral、pulse→neutral、plot/frontend→Workbench、storage→任何实验域，以及 leaf ui→Workbench。跨 owner value 的编码必须调用 owner projector/parser，不复制字段表。

不可破的 owner 规则：

1. `zlc_data`只拥有权威Dataset/Point/Validity/value数据事实；它不知道设备、Logic Node、Qt、Matplotlib、PlotSpec或Fit。旧`zlc_data.fit_*`及display-facing projection/transform模型在最后消费者迁走时删除；只有被neutral/FormalPulseScan真实消费的显式权威signal select/reduce纯函数可保留，且不认识plot kind。
2. `zlc_plot` 独占 Curve/Image/Histogram/Rolling/FacetGrid/PulseTimeline、数据投影、selector、Fit、overlay、Matplotlib persistent artists/blit、image decimation、Divider/固定 data box、size/DPR、style、raster worker、Qt front、headless export与live presentation。TaskConsole、Calibration report、DataFigure、FigureViewer、Edit snapshot与Pulse preview只能创建或消费同一`zlc_plot` session/spec，不能再手写第二套composer、geometry、selector、Fit或style。
3. 外部接纳源附带的同名 public `zlc_data` 不进入本仓。`zlc_plot` 的 public入口直接接收当前权威 snapshot，包内唯一私有adapter只构造只读plot view：保留完整`(R,P,*data_shape)`、dtype、PointTable/GridTopology、unit与validity，尽量共享源内存；它不是第二个public schema、artifact或Signal owner。
4. `zlc_frontend` 只拥有共享 Fluent widgets、typed form renderer、Qt application/window shell和把`zlc_plot.parameter_controls()`投影成Fluent控件的薄层。它不拥有PlotSpec、projection、selector、Fit、renderer、style或第二个Figure lifecycle。
5. `zlc_neutral_atom`拥有设备能力、领域Logic Node、Run、signal transaction与artifact lineage；它不拥有通用Figure，也不解析pulse backend私有字典。普通leaf可只导入headless-safe`zlc_plot.kinds/specs`声明推荐PlotSpec；不能导入session/render/backend。确有特殊Task preview/report时，只能通过lazy leaf UI contribution使用完整`zlc_plot` session，不把renderer带入headless declaration。
6. `zlc_pulse` 拥有pulse文档、编译、部署几何、typed execution observation和remote transport；它不反向导入neutral。
7. `zlc_workbench` 只拥有Qt窗口、卡片、signal路由、surface生命周期、同shot board present、cancel/close和产品布局；不得重新定义领域字段、Dataset shape、plot projection、Fit算法、执行后端或renderer。Qt callback和live publication路径不得遍历、复制、编码或hash大ndarray。
8. `Zou_lab_control` 保留为脚本、notebook和desktop共用的稳定API与composition root；它只做生命周期和窄委托，不实现领域算法或Qt/Fit状态机。
9. 新增或删除内建Device或Logic Node只改对应leaf。fixed-namespace discovery自动发现，不存在第二份中央concrete import/installation列表、service locator或mutable registry。

叶包边界也固定：一个`LogicNodeDefinition(key, kind, title)`表达三类节点的共同身份；每个fixed-namespace leaf只导出一个inert `LogicNodeDescriptor`。descriptor声明默认表单、device capability requirements、typed inputs/outputs、真实resource claims、一个domain bind/execute seam、Task-only preview与真正optional lazy UI。普通device choice由capability与stable instance id机械派生；role只作显示，不进入identity或binding。普通UI完全由通用descriptor projector生成。确实无法由声明表达的可选`ui/**`只导出lazy contribution；headless只验证它属于本leaf namespace，不加载Qt；Workbench启动时再通过frontend-owned generic UI context和同一`zlc_plot` surface实例化。leaf UI不导入Workbench、catalog或service graph，失败则该product启动失败而不是静默回退。

Request binding也只有一个owner：唯一Logic Node host按descriptor统一freeze authored draft并解析DeviceRef、SignalRef、artifact ref与普通fields；leaf的单一domain seam只构造真实领域request并执行acquire/compute/Port调用。request-dependent output cardinality等真实物理事实可以保留纯领域projector，但Package式`bind_api/prepare_hosted/bind_hosted_request/start_prepared/api_dependencies/dynamic_choice_fact/close_api`回调表禁止存在。public `Experiment.nodes`对所有leaf投影同一个轻量`NodeApi`；复杂领域便利函数可以由descriptor贡献普通函数，但不能再形成每leaf一套状态化API、Prepared/Bound阶段或repository/lifecycle。

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
- Selection 同步裁剪mask；Reduction按显式`ALL_REQUIRED/ANY_VALID/MIN_COUNT(n)`等policy计算输出validity；Fit逐batch排除无效observations并保留per-cell failure；Histogram记录dropped count；renderer显示invalid而不回退其它component。领域算法若有更强物理规则，由该leaf在通用validity之上声明，不能绕过它。
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
- PointColumn 的`value_kind`纳入typed codec并显式声明NUMERIC或TEXT；除`None`外不得混合数值与字符串，TEXT不得有unit。`None`是missing coordinate，不是一个可facet/group的类别：用到该coordinate的display会把对应row标为invalid并显示dropped count，Fit排除并记录observation count，GridTopology domain/映射则直接拒绝`None`。
- `AxisRoleId` 在这里只是语义标签，不把 PointColumn 变成独立 ndarray axis。合法 point-domain role 明确限定为 `SCAN_POINT/READOUT_EVENT/SPATIAL_X/SPATIAL_Y/SPECTRAL/HISTOGRAM_BIN/SITE`；`MONITOR_HISTORY`不再是Dataset role，`REPEAT/SCALAR/COMPONENT`也不得进入 PointColumn。monitor history只存在于Rolling PlotSession的私有小样本序列，不能由producer编码进P。未来若新增合法 role，必须先在唯一数据合同中扩展该闭集及codec test，不能由leaf自行放宽。
- `coordinate_id`复用已有`AxisId`，不再增加只在point column中改名转发的`PointCoordinateId`。它在一个PointTable内唯一；每列长度必须等于P；PointTable、GridTopology和cell schema直接属于同一个typed DatasetSchema，不计算普通内容fingerprint。
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
- 只有把point rows映射成二维/多维logical cells（densify、grid image、topology-dimension selector/facet）才需要GridTopology；普通`PlotKind.FACET_GRID`是“一个facet source的small multiples”容器，可facet R、trailing data axis、PointRows（逐ordinal）或PointCoordinate，完全不要求GridTopology。
- 旧`AxisLayout/FitResultBatch`不再服务绘图或Fit；FacetGrid batch结果只使用§4定义的`FacetFitBatchResult`和现有FacetData cell顺序。

### 3.5 Dataset 与 zlc_plot 的唯一接缝

权威数据仍是当前 `zlc_data.OwnedSnapshot`（或其最终同义单一值）：schema、values、validity、revision 与 PointTable/GridTopology 只在这里保存。纳入仓库的 `zlc_plot` public API 直接接收该权威 snapshot；包内唯一私有 adapter 可构造只读 plot view，但必须满足：

- values 与 source 尽量共享内存，保持原 dtype；不得转成完整 float64、复制整张相机图、展开 H×W coordinate grids 或创建同尺寸 bool validity；
- 完整保留 `(R,P,*data_shape)`、scalar carrier `(1)`、point authored order、component validity、unit 与可选 GridTopology；
- adapter 不导出 public Dataset/schema，不持 signal lifecycle、shot/provenance 或 artifact identity，也不计算 payload/schema/transform SHA；
- unit只从zlc_data读取canonical string；zlc_plot内部可做显示换算，未知unit按opaque identity处理，不导入外部UnitRegistry或建立量纲代数；
- `P=1` 且无 coordinate column、普通非 grid point list、重复/非单调 coordinate、sparse/serpentine grid 都必须原样可画。

`zlc_plot.AxisRef` 是 presentation 中唯一轴引用词汇，直接映射权威数据事实：

| AxisRef | 权威来源 | 语义 |
|---|---|---|
| repeat | Dataset R | acquisition sweep；不是 pulse RepeatRegion 或 monitor history |
| point rows | authored ordinal `0..P-1` | 普通非 grid sequence/sample/facet |
| point coordinate | PointTable numeric/text column | 相关坐标；重复值不产生 Cartesian product |
| point dimension | GridTopology declared dimension | 只有 producer 明确声明 topology 时合法 |
| data | trailing data axis | 任意多维 data_shape 的真实轴 |

所有 Curve/Image/Histogram/Rolling/FacetGrid 的 selection、reduction、sample、facet 与显示状态由 `zlc_plot.PlotSpec/PlotSession` 唯一拥有。不得在 `zlc_data`、frontend、Workbench、Calibration 或 FigureViewer 中保留第二套 `ViewSpec`、`CommittedTransform`、projection DTO、kind-specific selector 或 fit translator。

投影前必须完成一次**轴覆盖检查**。覆盖单元是R、每条trailing data axis，以及P row domain；一旦使用`point_dimension`，producer声明的每条GridTopology dimension成为P上的独立逻辑覆盖单元。每个单元必须由plot coordinate、group、facet、Histogram显式sample、PlotSession显式index/filter或R专用reducer明确处理，同一逻辑单元不能同时被固定index后又sample/reduce。无GridTopology时，多个PointTable coordinate只是同一P rows的相关标签，可共同定义coordinate/group/facet/filter，绝不能被当成多个独立tensor维；有GridTopology时，`sample(x)+index(y)`等不同逻辑维组合合法。resolver补完display index后仍未覆盖的P/grid/data单元使投影进入`NEEDS_INPUT`，不能落入`reshape(-1)`、`_all_positions()`或“reduce remaining dimensions”。

- Curve/Image的`reduction`只允许合并未被保留的R观测，以及映射到同一已画坐标的重复P rows；它永远不能折叠未选择的data axis。
- `HistogramPlot.samples: tuple[AxisRef, ...]`显式列出可pool/flatten的物理样本轴；未列出的轴必须facet或index-select。空或默认samples不具有“全数组flatten”含义。
- Image的两个轴若都来自物理P，只有两个不同的`point_dimension`且producer提供对应GridTopology时才合法；两个普通PointTable coordinate不能凭数值组合被densify成图。
- 未消费的非repeat轴size为1时index 0只是唯一元素的恒等选择；size>1时可用稳定index 0形成**仅显示层**的初始视图，但当前axis/index必须在Setting和图面描述中可见、可改。它不进入Scan/Processor authority。x/y/facet/sample角色本身存在多个同优先级候选时仍必须`NEEDS_INPUT`，不能用index 0替用户猜角色。
- 上述index选择直接住在既有`PlotSession DisplayState/ParameterControl`中，名字与choices由schema动态产生；不新增public `AxisSelection/ViewState/ProjectionResult`类型。

`zlc_plot` 提供一个按 schema/role 推导默认 `PlotSpec` 的纯函数；它只决定非破坏、可见、可改的初始视图：

- R 对 Curve/Image默认mean；Histogram只有在resolver明确把R写入`samples`时才pool；Rolling history不是R；
- 有信息的非 repeat data axis 默认 select/facet或要求输入，绝不静默 mean/sum；
- 普通 point sequence 可用 point ordinal 作 X；只有 declared GridTopology 才使用 point dimension；
- x/y/facet/sample角色存在多个合法候选时返回需要用户选择，不按 rank、singleton、长度、名字或当前值破平局；
- `AUTO` 只表示 resolver 尚未确定，永远不进入 numeric widget 或持久化为数值。

Scan/Measurement authority 永远不从 PlotSpec 反推。用户点击 Fit 时，`zlc_plot` 从当前 session 的 source authority、PlotSpec、facet、selection 与 fit model 一次冻结自己的 `FitSelection`：snapshot plot冻结一个exact source revision，Rolling冻结实际选中历史样本的有序source revisions。Fit结果不反向修改数据或显示合同。Workbench只保存 `plot revision -> exact SignalPublication/EventRef` 的小映射，使selector/Fit派生信号携全部exact parents，不能按signal name/latest重建。

FacetGrid 是一个普通 cell PlotSpec 的 small-multiples 容器，不是第二 renderer family。facet 可来自R、point rows、point coordinate、point dimension或data axis。Fit scope必须支持当前cell与all facets；all-facets复用同一projection/solver，返回唯一 batch result（facet address/axis metadata + parameter arrays + per-cell success），并由同一renderer在各cell画overlay。不得为Grid保留旧Fit实现。
## 4. Repeat、PlotSession、selector 与 Fit 的唯一终态

四个相似词必须保持正交：Pulse `RepeatRegion` 是单个 point 内的硬件 timeline loop；acquisition sweep count 才形成 Dataset R；plot 中的 repeat 选择/聚合只改变显示；monitor rolling history 是独立时间序列，既不是 R，也不进入 Camera signal shape。

绘图侧只使用接纳后的 `zlc_plot` vocabulary：

- `PlotSpec` 是 immutable 的 kind、AxisRef、Histogram sample axes、reduction、facet 与 labels 语义；
- `PlotSession` 是当前immutable PlotSpec、display state、persistent artists、selector、Fit、overlay、resize/DPR、live update 与 export 的唯一 mutable owner；语义变更只通过其原子的`replace_spec(new_spec)`完成；
- `RasterFront` 是可呈现的 immutable raster/geometry front；
- `ParameterControl` 是唯一 toolkit-neutral 设置描述，`zlc_frontend` 只把它映射成 Fluent 控件。

不得保留或重建旧 `ViewSpec`、`ViewContract`、`CommittedTransform`、`FigureIntent/FigureFront`、`PlotPanelSession` 或 per-kind projection family。`PlotSession` 内部可以用私有值冻结一次操作所需的 source/spec/selection，但这些值不能成为第二套公开 Figure 或 Dataset 合同。

### 4.1 默认投影与 repeat/facet

`zlc_plot` 内一个纯 resolver 根据权威 schema、declared axis role 和请求 kind 产生初始 `PlotSpec`：

- Curve/Image的R默认mean；Histogram只pool其`HistogramPlot.samples`明确列出的轴；Rolling把每个成功promote用于显示的source revision投影成一个小样本并追加到PlotSession私有history；
- 任何有信息的非 repeat data axis 默认只能 select、group、facet或请求用户选择，绝不静默 mean/sum；
- 普通非 grid P 默认按 authored point ordinal/显式 coordinate；只有 producer 提供 GridTopology 时才可引用 point dimension；
- 同优先级存在多个合法轴时返回需要输入，不按 rank、singleton、长度、名称或当前数据值猜；
- `AUTO` 只表示尚未解析的 UI 状态，不能写入 numeric widget、PlotSpec 或 artifact。

Rolling的x轴是plot-owned的历史ordinal/age，不是`AxisRef.repeat()`，因此`RollingPlot`不再拥有x AxisRef。其source projection必须在可选group外得到每组一个scalar；capacity-one ingress可按§5的monitor-latest语义supersede尚未promote的完整revision，只有成功promote的revision才把小sample与source revision追加到私有history。这样不会排队或保存完整Dataset，也不会把已画点错认成另一revision；formal/exact采集仍由SignalPlane/collector保证，不能借Rolling补数据。源signal保持`(R,P,*data_shape)`，buffer length不进入R/P/data_shape。`window`定义当前保留并显示的历史样本数，是Rolling产品语义而不是全局软件memory budget；增大window不伪造已经淘汰的旧样本。Cross携一个source revision，interval/Fit携实际选中样本的有序source revisions，Workbench据此恢复全部exact parents。

PlotSession建立后，物理维覆盖检查为其schema动态追加index choices；PlotSpec仍只描述稳定的轴角色，具体index属于display state。index改变只重投影当前snapshot；reducer/group/facet/sample等PlotSpec语义改变走`replace_spec()`，在同一Qt surface内原子换spec、projection与所需artists，不能由Workbench镜像字段或重建panel。Rolling的index或spec改变必须同时清空旧history和revision映射，再从新语义的首个sample开始，绝不能把不同投影含义的旧点混在一条曲线上。`axis_choices()`与`parameter_controls()`来自同一resolver，Workbench不得再维护repeat pool/sample/facet枚举。

FacetGrid 只是 `FacetGridPlot(facet, cell)` small-multiples 容器，cell仍是同一个 Curve/Image/Histogram session。facet可来自R、point row、point coordinate、point dimension或data axis；一个grid只有一个facet authority。非grid多维point sequence不被densify，显式grid可保留sparse/serpentine cell与validity。

Setting 中的 repeat/facet 控件直接编辑 `PlotSpec` 或 session display state，不再保存 pool/sample/facet 的第二套 Workbench enum。显示选择不具备实验权威：Scan、Measurement、Processor request永远不从当前plot反推；Fit只在用户请求时读取当前session并冻结自己的exact selection。

### 4.2 Signal 与 presentation 的边界

neutral 的 SignalPlane 只拥有 `SignalPublication/SignalFront` 与 exact parents，不导入 zlc_plot。Workbench 在把一个 exact publication 提交给 `PlotSession` 时，只保存小型的 `plot data revision -> exact publication` 关联；Rolling按其当前window保留每个可交互历史样本的同类关联。Rolling Cross解析一个source revision，interval/Fit解析其实际选中样本的有序source revisions，Workbench将它们全部映射成exact parents（或已有等价EventSpan），绝不能只挂当前latest。不得复制数组、计算payload hash、按name/latest重建父级或创建per-revision presentation sidecar。

同一board所连接的continuous signals仍由SignalPlane提升一个coherent `SignalFront`。Workbench先把该front中的全部panel更新准备完成，再原子present完整board；某个plot投影失败只成为该attachment的nonfatal错误并保留上一完整board，不能杀死GUI tick，也不能让其它panel跨shot推进。Standalone Figure没有SignalPublication，因此可以selector/Fit/export，但不得伪造derived signal。

`RasterFront.source_revisions`是renderer实际画入当前front的有序source revision事实，不由Workbench根据“最近提交”猜测。每个Panel只保留三类小关联：仍pending的worker请求、最新已完成但尚待Qt接纳的worker front、以及当前可交互presented front（Rolling则为当前window内实际画出的revisions）；失败、supersede、过期成功、spec/source-generation替换和surface退役都必须同步释放其余关联。该集合必须有界于pending加可见窗口，不能成为per-revision presentation archive，也不能靠定时清理掩盖生命周期泄漏。

### 4.3 Selector

Cross/Area/interval/threshold/color-limit与PulseTimeline selection全部由 `zlc_plot.PlotSession` 的同一interaction engine、Divider/data-box coordinate transform和overlay painter实现；Curve、Image、Histogram与FacetGrid cell不得各写一套。press到release固定同一data revision和geometry；resize/DPR使该gesture取消并重新compose，不能用新geometry解释旧raster。drag overlay走persistent artist/blit或轻量Qt front，不能每个pointer event重画base，也不发布hover。

Workbench把已commit的selection result中的有序`source_revisions`映射为exact publications并交给唯一SignalPlane派生入口。Snapshot-backed Cross发布选中sample/cell的数据值，坐标只进provenance；Area把exact typed bounds交给zlc_data唯一权威selection纯函数，发布保持source dtype、未选axes与component validity的Dataset。Rolling Cross发布对应小样本；Rolling interval/Area由zlc_data把已选择的小样本物化成`(1,N,1)`scalar Dataset，P只表示此次选择的authored sample rows，并携全部exact parents，绝不恢复`MONITOR_HISTORY`或保存完整源frames。zlc_plot内部为交互生成的full mask/flat indices不得作为Signal payload传播或复制整图。active gesture不发布formal结果；selection shape/schema变化换generation。相同selection也可交给producer声明的`Selection -> ParameterPatch`映射预填Edit draft，但zlc_plot不配置设备。

### 4.4 Fit

Fit model、argument parser、selection packing、solver、overlay与live scheduling全部属于 `zlc_plot`。Fit曲线/点直接画在同一session；对外只发布具名参数，不发布拟合曲线数组。显式Fit请求冻结当前plot front、ordered source revisions、PlotSpec、facet、selector/viewport与model；完成时若当前base front不匹配就隐藏旧overlay并等待/计算当前结果，不增加`LAG`产品标志，也不把旧result伪装成当前shot。

Monitor Fit是同一live session的one-active + latest-pending计算，base持续前进，Fit不能隐式Hold/Pause；Edit/FigureViewer Fit是其自己的snapshot session，只冻结该surface。Qt线程只做O(1) submit和小结果安装。规则大图保持原dtype readonly view，坐标只用O(H+W)轴向量，ROI优先slice/view，禁止H×W meshgrid、多份full-size float64/validity副本与Python逐元素扫描；先profile并修复杂度，再决定thread/process。

FacetGrid支持focused cell与all-facets两种产品操作，但不复用现有`FitScope`（后者只表示selector/viewport/all的单次选区）。实现只增加`fit_all_facets: bool=False`和一个`FacetFitBatchResult`：它按现有FacetData cell顺序保存facet AxisRef/value、每cell的`FitResult|error`及source revision，并由同一solver/overlay path逐cell处理。不得建立通用batch layout家族、Grid专用Fit engine、renderer或preferences。

## 5. Signal transaction 的唯一终态

```text
ProducerTransactionId:
  producer identity + generation + monotonic publication sequence

SignalPublication:
  one immutable transaction id
  one atomic sibling-signal bundle
  exact source Dataset revision/provenance
  zero or more exact parent transaction ids

SignalFront:
  immutable set of mutually coherent SignalPublications
  explicit transaction-graph frontier, never name-based reconstruction
```

- producer 在发布时创建 transaction；retainer 只管理生命周期，不能事后按 signal name 猜 transaction。
- selector、Fit 和 processor 派生信号必须携带 exact parent transaction。
- neutral SignalPlane 是唯一 transaction mint/retain/publish owner，并暴露窄 `DerivedSignalPublisher` port；zlc_plot/frontend完全不知道 ProducerTransactionId/OwnedSignalPublication。active producer/processor generation在bind时冻结typed parent graph、expected sibling bundle、failure policy与promotion policy，cycle启动即拒绝。`COHERENT_CONTINUOUS`只用于每个parent都应产出的processor/已commit selector；user-triggered Fit等`EVENT_RESULT`保留exact parent但从不作为source frontier的必到成员。Workbench board只提交当前实际连接的continuous signal-id immutable set；“causal cut request”只是该语义的名字，不授权public class。neutral只取这些signals的parent closure，不让未显示旁支或event result阻塞。
- neutral 为该closure生成immutable `SignalFront`；它就是被提升的causal cut：source与所请求derived outputs到达同一frontier后才原子提升；pending时继续呈现上一完整front；atomic siblings永不拆开；无依赖producer独立前进。formal/exact链每个transaction都处理，gap/failure fatal；monitor latest链可以按声明supersede整个未完成candidate，但只能丢弃整组、绝不能把不同parent的部分拼起来。Workbench所有相关panels从同一个promoted front取值，因此不会出现raw N与ROI/processor N-1；derived失败时保留上一完整front并显式呈现该generation error。
- plot attachment进入topology前先纯验证并冻结generation-static PlotSpec或resolver输入；运行中每个revision不再prepare或配对presentation sidecar。Workbench只从neutral提升的同一个exact SignalFront更新全部PlotSession并原子安装完整board raster front；pointer gesture从press到release固定对应immutable publication和plot revision。Area/Cross等已commit selector route各自发布COHERENT_CONTINUOUS，Fit参数单独发布EVENT_RESULT，不能把它们塞进一个动态“panel outputs”bundle使任一事件阻塞continuous front。异步Fit只在zlc_plot内部强引用其source view，Workbench以revision关联取回exact SignalPublication并通过DerivedSignalPublisher发布参数bundle。
- `EmissionSlotId`（nonempty canonical string typed id）与`PointEmissionMap`都由`zlc_data`拥有，neutral只消费。formal finite consumer的publication cardinality必须在bind/preflight冻结：`ExactlyOnePublication`明确表示每个`(repeat, base_point_ordinal)`恰好一个atomic publication；`FixedPublicationEvents(ordered tuple[EmissionSlotId,...])`表示每个base cell恰好K>=2个有序publications（K=1 canonicalize为ExactlyOne），slot ids唯一。后者由collector在reservation前生成`PointEmissionMap(base_point_count, ordered slots, expanded_row -> base_ordinal+slot_ordinal)`，把base PointTable每行按slot顺序复制，并追加一个由consumer-binding id派生、冲突即拒绝、role=`READOUT_EVENT`、TEXT的emission-slot PointColumn。若base有GridTopology，则expanded topology在现有`dimension_ids/coordinate_domains`tuple末尾追加该emission-slot AxisId及domain，row_to_cell追加slot ordinal，保持injective；不得建立GridDimension wrapper或把K rows重复映射到同一base cell。expanded table/topology/map直接作为typed schema/artifact facts保存，不计算普通payload fingerprint；runtime不得观察到K后再改schema。
- fixed-event collector 精确 reserve `R*baseP*K`，按 `(repeat, base_ordinal, slot_ordinal)` 消费；missing、surplus、错序、generation change 全 fatal。一次 SignalPublication bundle 内的多个 sibling outputs 或一个 Value 的多个 data components 均不算多 publication event，不触发 PointEmissionMap。
- “有 exact parent”不等于“可当 FormalPulseScan y”。neutral-owned `FormalAssociationCapability` 是唯一判据：纯 one-to-one、固定配置的 selector/projection/processor 保留 upstream association并声明 ExactlyOne；固定 fan-out 必须在 bind 时声明 FixedPublicationEvents并携 PointEmissionMap；fan-in、跨事件 Fit、Histogram/Distribution 或任意无法逐事件重放的变换明确无 capability。zlc_plot只给出已commit selection的纯变换事实，composition/neutral根据leaf声明铸造capability；TaskConsole不递归猜。selector配置改变会换generation，正在bind的scan必须失败而不是沿用新ROI。
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

硬件lease的释放边界早于Run terminal：`hardware -> cleanup/DISARM/SAFE + device-buffer-sealed receipt -> revoke capability -> release lease exactly once -> domain validate/write/report -> terminal state`。receipt必须同时证明SDK/DMA/ring buffer借用全部结束，post-safety输入已是进程自有immutable data；不能把会被下一次arm复用改写的view带过release。post-safety阶段无硬件能力；领域编码、写入或报告再慢也不能让安全空闲设备继续busy。

## 6. 运行、硬件、Frontend 与领域纵切合同

### 6.1 Task、Measurement、Processor 与 Fit

- Task 是一次用户 use case，可顺序组合少量 flat Runs；它不是递归 workflow、child plan 或全局调度器。
- Measurement 声明采集的物理语义和 output vocabulary；live/finite 只是同一 Measurement 的 host policy。Camera 只有一个 Measurement，具体使用 qCMOS 或 MOT Camera 由 typed device role/binding 决定。
- Processor 是 typed input 到 typed output 的领域变换。latest-only、exact、finite 是 delivery/execution policy，不产生新的 Processor catalog、form、binding 或 lifecycle。
- Fit 的model、packing、solver、live scheduling、overlay和参数结果都属于`zlc_plot`。Fit不是neutral通用Processor，也不取得上游设备；Workbench只把具名参数作为exact-parent EVENT_RESULT路由到SignalPlane。
- selector 属于PlotSession。Area发布保留dtype、axes、validity和exact selection provenance的Dataset；Cross发布所选sample/cell的数据值，坐标只进入provenance。Fit只发布具名参数sibling bundle，不发布拟合曲线数组；拟合曲线、点、中心或半径作为同一zlc_plot overlay可见。
- 普通 pointer motion 不发布 hover 数据。Area/Cross/Fit 不重配 Measurement、不建立 ROI Measurement/Processor，也不弹出第二个 DataFigure。

通用Logic Node host唯一拥有start/stop、Run观察、input cursor、same-shot atomic publish、request binding、通用form、API forwarding、普通持久化与错误投影。现有`HostedRun`必须原位演化或一次性重命名为这一唯一host；`HostedProcessor`和`LiveDatasetHost`的职责折入后删除，不能旁建第二host。host内部允许两种真实策略：Task/Measurement执行flat `RunPlan`；Processor由host持event cursor并调用leaf纯`compute(source, request)`。策略不得泄漏成两套公共lifecycle类型或TaskConsole分支。

普通leaf只保留一个discovered descriptor、一个领域request/value、typed inputs/outputs、真实resource claims及acquire/compute算法；Camera live/finite复用同一个acquisition path，同类scan Measurement复用同一个scan collector。不得为每个leaf重复Intent/Request/Bound/Prepared/API/presenter/codec/repository/lifecycle链；只有独立不变量或第二真实消费者才能保留额外类型。Calibration capture→analysis这类Task明文要求的多个flat Runs是Task的领域步骤，不是可复制到普通leaf的阶段骨架。普通artifact由host拥有project目录与arrays-first/record-last提交顺序，leaf只提供少量可读record字段、科学arrays与reader；不得恢复per-leaf repository、canonical bytes或recursive externalizer。

普通fields、choices、dynamic outputs与推荐PlotSpec都由inert declaration提供；Workbench不按DefinitionKey写具名分支。新增普通Measurement/Processor的领域代码应为几十到低几百行；超出时必须指出不可替代的物理算法或特殊交互，而不能把通用骨架搬进leaf。

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
-> domain validation / direct write / report
-> terminal publication
```

terminal 只能在 worker、session、interrupt 和 cleanup 全部真实退出后发布。cleanup failure 使本 Run 失败并保留 primary/cleanup diagnostics，但不制造跨连接 quarantine 或持久门禁。新连接只凭实时 identity、当前 SAFE 初始化和当前 capability 建立 authority。

cleanup acknowledgement 只保存 exact session/binding identity 与 `source_stopped/no_more_work/joined` 等真实 terminal facts；无人消费的 acknowledgement/interrupt digest 不构成额外证明。`PhysicalDeviceIdentity` 只由可读稳定 endpoint/serial identity、evidence kind 与唯一 installation asset revision 组成，不把同一部署配置重复 hash 成 expected/evidence identity。

有限采集按冻结的 expected cardinality 完整配置设备 buffer。continuous monitor的source Dataset每次只表达当前publication；`RollingPlot.window`定义PlotSession私有保留/显示样本数，它既不进入Dataset cardinality，也不是全局软件内存预算。formal/exact事件不得因软件队列策略丢弃；monitor可以按明确latest语义整体supersede尚未开始且已经没有产品价值的candidate。

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
- 两者都绑定exact request/adapter identity、不可跨generation复用；FIRE后的terminal/count/stamp reconciliation必须匹配该边界。token缺失、重复消费、identity变化或deadline越界都在领域record原子发布前使Run失败。
- association token只在本次执行中拥有物理arm/bind/reconcile生命周期；finish成功即消费并丢弃。领域record保存已经存在的compiled execution、typed terminal、ordered `EventRef`与direct parents，不再把同一request/terminal编码成`SignalAssociationEvidence`、递归upstream evidence或其SHA让producer自证一次。

正常 Pulse 与 SCAN_SLOT/MOT 执行只使用现有 bitstream 的 autonomous streamed hardware timing。API-slot 无法无缝更新时才允许既有、显式标记的 segmented `STATIC_ONCE` 路径。不存在逐 cell host fire/wait、software sleep timing 或为了架构偏好新增 trigger FIFO/counter/ROM attestation 的 baseline。

FormalPulseScan 只拥有：

```text
frozen pulse program
+ sequencer Run
+ 已经运行的外部 Signal(y)
+ producer-owned FormalAssociationCapability
+ exact terminal/coverage evidence
```

它不取得 Camera、Processor 或 Figure 的设备，不启动/停止上游，也不按 producer 类型建立第二条 capture pipeline。普通 cursor 只证明软件交付顺序；正式 scan 必须在 FIRE 前冻结 publication cardinality/association，在 FIRE 后绑定 exact pulse terminal，并在领域record原子发布前验证每个 event、coverage、generation 和 lineage。

qCMOS 正式资格使用现有硬件能力，并明确区分“路径语义资格”与“本 Run 事实”：

1. 一次 arm 为外触发模式，并按整 Run 的冻结帧数配置；
2. E0 qualification 只主动证明当前 adapter/连接、ROI、binning、readout、trigger wiring/mode 与 counter/stamp 的 ordered one-frame-per-trigger 语义；不得由少量固定 trigger 虚构最大 scan count、最大 delivery latency或无限 sustained-delivery envelope；
3. 每个 Run 以 pulse artifact 为完整 schedule 的唯一 owner；association 只冻结常数级的 artifact/schedule fingerprint、channel、count、minimum spacing 与 clock，不能把 O(N) schedule arrays 再写入 lineage；
4. preflight 读取当下冻结 exposure/工作点的真实 minimum trigger interval并核对完整 compiled schedule 的最小间距；不满足则 FIRE 前拒绝。新 finite Capture 在 physical arm 时把设备 buffer 配成该 Run 的 exact cardinality。已经运行的 signal association 绝不重arm、resize或重配上游，而是在 FIRE 前冻结当前 arm/session 与 stable produced/drained/publication baseline，依靠上游持续排空有限 driver ring 和无损 raw stream/FollowTap 交付；ring overrun、少/多/乱序、stream publication failure或期限届满都使整 Run INVALID。它保证 fail-closed 正确性，不承诺任意长 schedule 在固定 ring 上必然成功。只改变 exposure 时，endpoint须证明其它 qualification-scope facts未变并用新 readback重新计算 timing/quiet facts；
5. Run 末端比较 expected/emitted/produced/observed/drained count，检查 frame/camera stamp 与 timestamp 的单调性，再经过由当前物理 trigger interval 派生的 quiet window确认counter不再增加；普通SDK timeout只作为本次失败期限，不能命名或持久化为“qualified maximum delivery latency”；
6. 任一缺帧、多帧、乱序、late extra、counter倒退/wrap歧义、generation change 或 terminal/coverage 不一致使整 Run INVALID，不能提交、不能自动重跑。

该保证是 preflight + per-run reconcile，不声称具备现有硬件没有的逐沿 tag。只有 E0 或代码证据证明现有 RTL 真 bug/偏离既定设计时，才单独评估与根因直接相关的最小硬件变更；任何硬件修改都不由本文自动授权。

### 6.4 zlc_plot、Frontend、render 与表单

绘图全栈唯一入口是 `zlc_plot.PlotSpec -> PlotSession -> Qt5PlotWidget/RasterFront/export`。TaskConsole、Calibration、Occupancy、DataFigure、FigureViewer、Edit snapshot与Pulse preview都使用这条入口；它们不能手写composer、projection、Divider、style、selector、Fit、raster或export。公开kind闭集严格沿用接纳源：`CURVE | IMAGE | HISTOGRAM | ROLLING | FACET_GRID | PULSE_TIMELINE`。SiteMap是Image加typed overlays，Distribution是Histogram的双高斯/threshold模式；Meter不是正式plot kind。Histogram必须显式声明sample axes；Rolling使用plot-private revision history而非Dataset R；两者都服从§3.5的物理维覆盖合同。

FacetGrid复用普通cell session，不拥有第二renderer、Fit或form。Saved Fit Grid只增加typed facet address与导航，所有cell raster/selector/refit/export仍走同一PlotSession。Calibration report的overview、SiteMap与Distribution也只是同一API的headless/Qt消费者，因此与TaskConsole同spec、style、size/DPR语义。Calibration可在自己的lazy `ui/**`保留一个薄adapter，只把已经声明并物化的FINAL Dataset/annotation映射成现有`PlotInput + PlotSpec`，且Qt与headless export共用这一个映射；它不得定义report Dataset/schema/DTO、第二materializer、renderer、style、Fit engine或通用Workbench report manager。

`zlc_plot`独占persistent Matplotlib artists、blit、image decimation、Divider/固定data box、coordinate mapper、font/color/style、logical size、DPR与raster worker。resize/DPR改变时旧front不stretch作为最终画面，而是同revision重compose；pointer drag、selector更新与Fit overlay优先更新既有artists/overlay，不重画base。canonical Figure字体资产只在`zlc_plot/assets`保存一份`Helvetica Light`并随package发布；Qt shell字体和QSS token仍由zlc_frontend统一使用Segoe UI。Calibration/leaf不得局部换font/style。

`zlc_plot.parameter_controls()`是display setting唯一描述。接纳源的通用`Qt5ParameterPanel`不进入产品public surface；`zlc_frontend`的唯一Fluent mapper将`ParameterControl`投影为：bool→`FluentSwitch`，numeric→`FluentSpinBox`，固定短choice→segmented/radio，动态/长/可扩展choice→`FluentComboBox`。None/AUTO/Select/Inherited/Unavailable不编码进numeric widget。普通用户面不暴露无意义的x/y min/max；Image的color range、colormap、interpolation和colorbar是正式kind参数。Setting、Edit、Viewer、Calibration与Grid cell不能再保存第二份form schema或handler。

producer参数表单也只有一个frontend typed-form mapper。Logic Edit与Plot Edit读取同一producer-owned draft/controller；Plot Edit不是另一个配置模型。conditional enable/visibility由同一schema presenter派生。普通Qt draft原位reconcile稳定widgets；Add/Remove/Reorder只增删移动对应子树，unit/name/value/delay/binding/visibility不重建整树。已经进入正式object tree的QWidget始终由最终pane拥有，删除用`hide()+deleteLater()`，不以`setParent(None)`或临时top-level完成重排。

新Logic-node row必须先由leaf declaration defaults形成完整typed draft，再进入signal topology、dynamic output推导或request构造。Logic tree与signal picker只有一个generic projector，只读显示`R×P×data_dim*`、dtype、unit、generation与formal-association；history/buffer不进入shape，标量显示`R×P×(1)`。任何leaf不得创建专用dimension row或可编辑shape字段。

GUI快轨固定为offscreen `ensure_qt_app()` + 正式composition/launcher widget + 真实Qt input + outer grab；慢轨从正式`.bat/.py`按人类流程运行。两者共享QApplication owner、style、sizing、data path和交互步骤。手工假QWidget、直接调controller、无文字截图或另一套DPI不能验收。

### 6.5 Live Fit、Snapshot Fit 与性能边界

Monitor/Setting Fit是PlotSession内的live operation，绝不隐式Hold/Pause。base持续呈现最新front；每个session至多一个active与一个latest-pending计算，多个session公平。结果只在exact source revision匹配时画overlay，失配即隐藏，不增加LAG状态控件。Fit参数作为`EVENT_RESULT`发布，不进入continuous causal frontier。

Edit/DataFigure Fit 是 snapshot-scoped invocation，只修改该 snapshot surface 的 overlay/result；它不读写 Monitor Fit state，不冻结任何 Monitor panel。显式 Pause、pointer gesture 与 Edit snapshot可以持有自己的 base，Fit 按钮本身不能。

Fit compute、base raster compose和selector Dataset materialization是PlotSession内部互不阻塞的operation；不是三个public lane/manager。Qt只O(1) submit readonly source/spec并接收小结果。规则H×W图像使用zlc_plot内部regular-raster path：保留原dtype readonly view，坐标用O(H+W) axis vectors，矩形ROI用slice/view，validity用紧凑mask/broadcast，seed/moment用浮点accumulator，score/solve按需要chunk；不得生成两张H×W coordinate grid、多套full-size float64 copy或Python逐元素检查。执行后端只在复杂度修正后的profiling证明需要时选择thread/process。

Fit源码按真实内部依赖拆分，而不是按产品入口复制：model/result、projection/packing、solver、overlay可保留各自模块，但只有`zlc_plot`一个public facade和一个Fit engine。TaskConsole、Calibration、DataFigure、FigureViewer与Grid不得再有fit文件或translator；被替代的`zlc_data.fit_*`和`zlc_frontend.fit_*`完整删除。

Distribution 的双高斯与 threshold 是 Histogram/Distribution owner 对冻结的**单一原始样本 series**运行的窄 display-only analysis；它独占 bins、双高斯求解、解有效性判定、threshold、曲线/阈值 overlay 与视觉样式。多 series 普通 histogram 只画分布，不能静默取第一条做自动分析。求解必须验证收敛、finite positive sigma/weight、ordered/separated components以及threshold落在有意义的数据域；失败时不画伪曲线/伪threshold，只显示统一诊断。显式 Figure Fit 存在时覆盖该显示分析。

Calibration只拥有物理标定算法和领域结果：原始每site样本、label/validity、最终runtime thresholds/fidelities以及SiteMap事实。报告把同一原始样本Dataset与实际runtime threshold annotation交给zlc_plot Histogram/Distribution；不得构造`population`轴、复制dark/bright样本、先减threshold、把零伪装成拟合阈值，或在leaf中另写bins/双高斯/projection/style。若没有独立物理意义明确的全局模型，就不生成pooled页。

### 6.6 Device graph 与 Device Manager

内建device type使用固定namespace、一个leaf descriptor和确定性discovery，不是plugin registry、entry point、service locator或中央concrete import表。每个leaf唯一声明`type_id/domain/authoring_schema/defaults/factory/capabilities/typed requirements`，以及确有真实设备枚举能力时的optional discover；新增/删除类型只改该leaf。requirement字段是该type参数schema中的stable-instance-id引用，不以可改role或Python类名作连接；leaf声明每个引用要求的capability，composition据此建立依赖边、检查missing/wrong-capability/ambiguous/cardinality与cycle。它不是另一份graph DTO或mutable registry。

Installation文档是一个ordered heterogeneous DeviceInstance tuple。每个实例只保存stable instance id、role、type id和该type的parameters；role可改但不是identity，跨设备参数引用只保存目标stable instance id。Virtual/Remote/Hardware只是创建该graph的预填模板或Port实现，不能再拥有整套设备字段。composition按descriptor requirement与实际capability解析typed binding，missing/ambiguous/wrong-capability/duplicate stable id或role与依赖cycle在Experiment可用前失败。

factory只向通用runtime发布该实例实际提供的`capability token -> narrow Port/fact`；domain只用于分类与默认选择，绝不暗示“所有camera都有monitor”等能力。runtime只保留一个按stable instance id解析的capability表和一个通用`require_capability(DeviceRef, token)`入口，不得中央导入或维护Camera/Pulse/RF等具体Port字典/accessor。`DeviceRef/DeviceInfo`以stable instance id钉住runtime设备，并投影role、type id与capabilities；role只作人类绑定名。这样新增只有finite capture的camera或新domain时不修改runtime、public facade或Device Manager。

Device Manager只编辑该graph：按domain分组、Add device、每卡独立role/type/type-specific form、Remove/Retype、New/Load/Save/Apply，并把discovered hardware与loaded session分开显示。requirement字段由descriptor声明的capability实时投影为当前graph内兼容stable instance id的下拉选择，不让用户填写role或Python类型；缺失/不兼容的旧引用只作为明确的待修复项显示，不能通过Save/Apply。普通字段/role编辑原位更新；Add只插一张卡，Remove只删目标卡，Retype只替换该卡参数body，New/Load按stable id reconcile，任何操作都不全量重建卡片。Apply是application topology replacement：存在active Run时明确拒绝；idle时先做无副作用的完整graph validation，再由旧Experiment正常close→SAFE→release，随后连接新graph并只在成功后发布为active。新连接失败时用previous frozen graph恢复旧Experiment；若恢复也失败则明确进入no-active-session错误，绝不并存两套owner或把失败graph标成active。public `Experiment`对象身份在成功替换或成功恢复后保持不变；composition root退役除Device Manager外缓存旧runtime选择/Port的Workbench handle，之后由同一Experiment按需重开，不能让旧窗口继续持有上一代capability。正式device参数来自leaf schema，不能反射driver`__init__`，不能把qCMOS/Basler/sequencer/readout参数重新混进一个backend-wide form。它是desktop默认Experiment composition入口；TaskConsole、PulseGUI与其它窗口共享同一Experiment owner。

### 6.7 Logic Node、Task takeover 与 Edit

Logic Node同样使用固定namespace与一个discovered leaf descriptor。descriptor只含单一`LogicNodeDefinition`、AuthoringSchema、typed inputs/outputs、capability-derived device requirements、actual claims、一个domain bind/execute seam、optional task preview与可选UI contribution。Task preview直接携`PlotKind | PlotSpec`，只允许Task声明；运行时始终传typed value，codec只存在于TaskConsole layout文件I/O，不能恢复string kind/params接缝。唯一host私有完成build/start/evaluate投影，并生成form、同构public`exp.nodes.*` NodeApi、start/stop/run、publish、record-last persistence和统一observation；不再保留两级callback package、per-leaf API/Prepared/Bound forwarding zoo、中央concrete import或中央字符串fact service locator。

一个稳定device instance字段可以声明多项必需capability以及只在该型号存在时才使用的显式optional capability；Device form只列满足全部必需capability的实例，leaf只能通过typed application context取得已声明成员。Camera finite/monitor因此共用同一个`camera_instance_id`，association可选能力缺失时不排除普通Camera；不得为每项capability制造第二设备字段，也不得让leaf绕过context访问runtime。

普通leaf没有专用TaskConsole form/presenter/binding。只有declaration + zlc_plot无法表达的真实交互才允许leaf-local lazy`ui/**`；当前baseline仅有PulseScan scan-table/slot editor与Calibration report workflow。它们仍复用frontend form和zlc_plot，不得导入Workbench或自画Figure；“字段较多”不构成custom UI理由。准入不是保留理由：没有生产producer/consumer的Occupancy artifact/load/cell navigator整条删除，Occupancy只作live Processor并发布typed Dataset siblings。

custom editor拥有的结构化值仍必须作为`AuthoringSchema`中的JSON-like `structured`字段进入同一draft、layout与request freeze，不能藏在Workbench sidecar。frontend通用form projector只省略这类没有普通Fluent控件的字段；TaskConsole必须找到descriptor声明的唯一custom form来读写完整draft，否则明确失败，不能静默丢字段或退回文本编码。

Measurement/Processor保持row-local。只有Task启动后由TaskConsole的唯一command-admission投影接管整个窗口：固定header显示task名、progress、stage与唯一Stop，所有其它mutating命令禁用；完成/失败/停止走同一清理出口。该投影直接消费既有HostedRun/RunSnapshot（只补真实completed/total facts），不建立第二Task session/controller。Occupancy启动只发布signals，不自动开panel；只有Task descriptor显式声明optional preview时才创建临时panel，普通recommended plot没有自动开图语义。

Logic Edit和Plot Edit是同一producer-owned typed draft/controller的两个视图。selector只产出物理Selection；leaf可选声明一个窄`Selection -> ParameterPatch`纯映射，例如Camera Area预填ROI、1D interval预填scan range。该patch只描述目标stable device instance与参数草案，由composition root转交已打开的Device Manager draft；Device Manager原位校验/更新对应卡片，用户仍须点击普通Apply，GUI不直接配置硬件，也不把ROI塞进Measurement request。zoom只改PlotSession视图；整个流程不维护第二参数truth。

TaskConsole只消费descriptor catalog、唯一host factory、SignalPlane与project paths；它保存authored values/input refs并投影同一个host observation，不拥有ConsoleNodeSpec、run/processor attachment、resolved/bound多阶段DTO、artifact resolver callback、output presentation镜像或按host类型分开的poll/error逻辑。Workbench选择只冻结稳定Signal/Artifact ref；host一次解析为typed runtime fact，leaf看不到Workbench row、producer request、run node或Camera专用binding。

`Zou_lab_control`是脚本、notebook与desktop共用的唯一public application API，暴露稳定`Experiment`、`exp.nodes.*`、device/application lifecycle与Workbench opener；它只做discovery结果投影和窄委托，不显式import具体device/Logic Node，不拥有领域schema、算法、plot或第二runtime。

### 6.8 关键领域纵切

- Camera Measurement：一个节点选择qCMOS或MOT Camera；live固定`(1,1,*frame_shape)`，finite的每个`frame_i`从progress到FINAL固定`(K,1,*frame_shape)`，未完成cycle只由validity表示；history不进shape，uint8/uint16原dtype保留，同cycle多帧为atomic siblings。
- Calibration：内建readout task，不是plugin。capture与calibrate是两个linked flat Runs，显式`save_frames`；live与saved frames走同一算法。成功后Experiment只保存可见可改的`current_calibration_ref`默认指针；artifact/report写入§7的task目录。报告只消费普通FINAL outputs和zlc_plot PlotSession；SiteMap物理事实与Calibration artifact仍属neutral。默认runtime model的`readout_samples`严格是`(source R, non-event context P, SITE)`，SITE不搬进P；当前CalibrationReport只保留context axis identity/index order而不保留source coordinate scalars，因此该输出使用bare PointTable authored rows，绝不能在UI按context index伪造物理coordinate columns。
- Occupancy：request冻结显式CalibrationArtifactRef，current ref只作一次预填。它消费same-shot frame/calibration facts并原子发布typed siblings；不会自动开panel。
- MOT Field：Ready→Running→multiple live→FINAL可观察；point rows保持authored order，标量shape为`(R,P,1)`，真实7×7×7只在GridTopology表达，accumulator不得O(N²)复制。
- PulseScan：消费任意已经运行且有FormalAssociationCapability的y signal；不绑Camera。sweep count形成R，Pulse RepeatRegion不改变R，P来自冻结PointTable，非grid trajectory保持普通rows。
- Area/Cross与Processor连续输出进入同一exact parent graph；Fit参数是EVENT_RESULT。ROI→ROI→Fit和多份仍在使用的revision不能按name/latest重建。

每个领域切片必须从正式Experiment/SignalPlane/zlc_plot产品入口验证，不以单元算法、synthetic renderer或旧测试适配替代产品流。

## 7. 项目输出、持久化与 identity 的唯一终态

composition从用户显式选择的project root创建immutable WorkspacePaths；默认task/run/figure roots就是该project下可见的`tasks/`、`runs/`和`figures/`。leaf只得到自己的窄Path；不得从package位置、home、CWD或环境变量猜输出，不得写死源码包、`_output`或隐藏repository。deployed geometry、target manifest、RTL/bitstream等immutable部署资产仍由package resource或显式deployment config拥有。

Artifact 是实验产品文件，不是内容寻址缓存。每个领域owner在可读目录中直接写自己的文件：

```text
project_root/
  tasks/<task-kind>/<run-name>/
  runs/<measurement-or-processor>/<run-name>/
  figures/...
```

`run-name`由领域owner生成一次可读且唯一的名字（时间/短序号/用户label），typed ref保存相对project的领域路径和artifact kind；它不是payload digest。输出只保留用户会读取/重载的领域record、少量科学arrays与明确请求的PNG/CSV。Calibration raw frames只由`save_frames`控制；不得为每个中间值拆文件、复制capture metadata或生成size manifest。大array按原dtype写`.npy`或领域明确格式，小metadata使用普通JSON/CSV。每个文件先写同目录临时文件并`os.replace`；多文件artifact最后原子发布唯一领域record（如`calibration.json`），不引入CAS、blob pool、generic manifest、dedup、commit journal、pending inspection或repository lease。

硬件生命周期与磁盘写入分离：

```text
hardware terminal -> validate physical/cardinality facts
 -> cleanup/SAFE/join -> revoke capability -> release hardware ResourceLease once
 -> encode/write domain files -> atomic publish domain record -> typed path ref
```

磁盘或报告失败只影响对应Run/artifact结果，不重新占用硬件，也不产生quarantine或跨重启协调器。若领域已经在FINAL machine artifact后才生成可选operator report/frame export，则失败作为可见warning，不回滚FINAL artifact。

Identity严格按成本和消费者限定：

1. live frame、Dataset/Value payload、EventRef/EventSpan、selector/ROI、Plot/raster、Fit input/result、普通config/artifact与runtime provenance禁止完整payload SHA、canonical tagged bytes或逐元素hash；exact generation/sequence/transaction/source ref、typed schema value与path已足够。
2. 允许保留的fingerprint/digest只在真实FPGA/transport比较边界：现有pulse document/compiled artifact/deployed geometry/target/bitstream/wire/ABI握手。它们不得扫描live ndarray，也不得泛化为普通artifact identity；普通schema evolution使用可读版本字段而不是内容hash。
3. provenance记录真实typed parent refs、run id、generation/sequence、algorithm/model id与参数；不能再派生一串content/reference/span/join digest来“证明”同一组事实。
4. `zlc_storage`只拥有path confinement、atomic file write/replace、目录durability和普通小metadata encoding；领域codec/validation仍由领域owner负责。`ContentAddressedStore`、普通content/config digest、`$zlc-bytes`、size manifest、repository lease、prepared commit与对应tests全部删除。跨进程**设备**互斥仍由设备owner的InterprocessDeviceLease负责，与artifact storage无关。
5. qCMOS/terminal/cardinality/lineage INVALID必须在领域record发布前判定；确定性合同错误不能被磁盘重试恢复成成功。

## 8. 依赖闭合的实现顺序

不得逐症状打补丁或提交双owner。每个cut都必须同时替换公开合同、全部生产/消费路径、相关测试/文档并删除旧owner；中间可在工作树内分步，但checkpoint/commit不能保留兼容层。当前PointTable、SignalPlane与冻结硬件合同只在真实依赖要求时复核，不重做已闭合工作。

### C0：规范、清单与 characterization 冻结

- 完成本文件、临时审查台账、固定LOC/public surface、zlc_plot接纳commit、文件排除表、旧owner删除表和产品证据矩阵。
- 接纳源永久记录为外仓`zlc_plot main@4fca73fcafc5b0a65a994399cf4641ed3b52bc8a`。只接纳其tracked`src/zlc_plot/**`，并显式排除`src/zlc_plot/qt_controls.py`、全部`src/zlc_data/**`、notebooks/PNG/build/egg-info/cache；保留`py.typed`和唯一Helvetica asset。该VCS commit id只是第三方源码provenance，不是运行时payload SHA机制。
- 接纳源本身不是不可修订规范：C0已证实其Curve/Image会聚合所有未引用轴、Histogram会flatten全部values、Rolling会把R当history。接纳时必须按§3.5/§4.1纠正，不能把这些行为带进本仓。
- 先对外部zlc_plot核心写characterization：六kind、selector、Fit、DPR/resize、live/raster/style/export/PulseTimeline、N-D/validity/non-grid/grid/dtype/zero-copy；另加物理维覆盖、Histogram显式samples、无topology双point-axis拒绝、Rolling私有history/monitor-latest coalescence/exact displayed revisions/spec-change reset，以及Camera monitor每次publication严格`(1,1,*frame)`。
- 不为历史tests保留alias；旧测试与owner在其cut同删同改。

### C1：zlc_plot 单一绘图全栈替换

- 按C0固定manifest导入tracked核心、font与必要package metadata；`qt_controls.py`文件本身和`Qt5ParameterPanel`导出均不进入本仓。
- 实现一个私有readonly Dataset adapter、物理维覆盖检查及`default_plot_spec/axis_choices`。对接纳源public surface只做必要修订：`HistogramPlot.samples`、移除`RollingPlot.x`并改为session-private revision history、`PlotSession.replace_spec()`、现有`SelectionData/FitSelection`增加ordered `source_revisions`、`fit_all_facets`与`FacetFitBatchResult`；不增加public axis-selection/history Dataset/DTO/lane/registry。
- `RasterFront`冻结实际画入的ordered source revisions；Workbench的revision→publication关联严格有界于pending/latest-worker/presented-window并在所有terminal/supersede/retire路径释放。Task preview在declaration到PanelConfig之间全程保持typed`PlotKind | PlotSpec`，只有layout I/O可codec。
- 在同一未提交cut迁移TaskConsole、Calibration、Occupancy、DataFigure、FigureViewer、Edit、Pulse preview与public facade；Camera monitor改为每publication只发布最新`(1,1,*frame)`，同cut删除`MONITOR_HISTORY` Dataset role、camera request/API/default/authoring中的`history_cycles`、capture-preview history point-column路径和`MonitorDataset.append_window`环形分支。`MonitorDataset`本身保留为通用stream materializer，并压成`keyed_cycle + latest_cell`两条真实模式；`latest_cell`继续唯一承担payload/metadata校验、ordered ingest、revision/event_ref/head、gap accounting、atomic replacement与snapshot publication，不能把这些职责移入Camera leaf或GUI。随后删除全部旧frontend plot/projection/selector/Fit/render/raster/style/layout、zlc_data Fit closure、Workbench/leaf专用composer及相关测试。任何旧plot import、history-in-P/`history_cycles`或第二runtime尚存都表示C1未完成。
- 用Camera→live Image→Area→第二Image→Fit、Histogram Distribution、Calibration report、FacetGrid batch Fit、FigureViewer与Pulse preview的正式快轨证明后才commit，并报告固定口径净删除。

### C2：DeviceInstance graph

- 原位改写Installation文档为ordered heterogeneous instances，建立leaf type descriptor/factory discovery、stable-id requirement graph和typed capability resolver；删除runtime内按具体Port类分栏的中央能力表。
- DeviceManager改为per-device cards与New/Load/Save/Apply；迁移public composition及所有device consumers。
- 同cut删除backend-wide config/package/plan/dispatch、flat editor、installation-config digest/CAS conflict链、中央concrete imports及旧tests；其它领域artifact storage不属于C2。以多device add/retype/save/load/apply和正式Experiment启动证明。

### C3：最小 Logic Node host

- 把三种Definition合成一个`LogicNodeDefinition`，每leaf改成一个discovered descriptor；把start/stop/cursor/publish/binding/form/API/persistence/error投影收进由现有HostedRun原位演化的唯一host，不新增阶段类族或并行host。删除Package callback table、中央字符串facts、HostedProcessor/LiveDatasetHost与TaskConsole attachment/spec/resolution第二胶水。
- 迁移全部leaf，保留真正领域算法/ports与三个已准入optional UI；每个leaf从本cut起使用§7的host-owned目录/record-last writer并只声明自己的可读record与科学arrays，删除普通SHA/bytes/repository以及重复API/Prepared/Bound/presenter/lifecycle、中央imports与具体Workbench特判。
- 每个leaf执行discovery、typed form、construct、start/stop virtual smoke；Camera、Processor、PulseScan等代表纵切执行正式产品流。报告每leaf旧/新LOC，普通leaf仍数千行则C3未完成。

### C4：Task takeover、Edit与输出

- 用既有HostedRun/RunSnapshot实现唯一Task takeover投影；恢复header/progress/stage/Stop，统一禁止其它mutation；Measurement/Processor保持row-local，Occupancy无auto-panel。
- Logic Edit/Plot Edit接同一producer draft，完成Selection→ParameterPatch与Apply路径。
- product composition暴露project下可见的`tasks/`、`runs/`、`figures/`，Calibration creation UI恢复`save_frames`；迁移尚未属于C3 leaf的Capture/Fit/Figure/application输出。当最后一个真实consumer迁走时，本cut唯一删除generic CAS/content store/tagged-bytes/size-manifest与对应tests，不重复删除C2/C3已经闭合的owner。
- 以Calibration、MOT、Occupancy、Camera Edit/ROI和冲突自动retire-then-start产品流证明。

### C5：PulseGUI narrow hold/step

- 建立replace-applied-pulse命令并复用现有compile/upload/prepare/fire seam；缓存同document revision points，原子更新held front。
- 删除完整Run cancel/reap/restart路径与其UI重投影；分别profile compile、RPC/upload、safe/reap、UI，保持RTL/Tcl/XDC/bitstream/wire零diff。
- 通过Virtual/Remote/Offline与真实server路径验证Stop/hold/step、dirty state、endpoint clamp和稳定文本。

### C6：领域E2E、性能与全仓清理

- 执行§9全部产品流、2304²性能、两条GUI证据、public Device/Logic Node枚举smoke和真机可执行runbook；不能用单元测试替代。
- 全仓逐文件change-impact复核，删除空目录、dead wrapper、compat alias、旧tests/tutorial/docs与临时台账，证明所有旧owner/普通SHA/软件预算/中央特判为零。
- 最后才跑broad current suite。失败只按仍有效物理/public合同处理，不恢复被删架构。

## 9. 最终验收门

只有同时满足以下条件才可声称迁移完成：

1. `zlc_plot`是唯一plot/projection/selector/Fit/render/raster/style owner；旧zlc_frontend/zlc_data/Workbench/leaf plot模块、imports、symbols、fonts、tests与目录为零。外部第二zlc_data与Qt5ParameterPanel未进入本仓，所有产品消费者只走public PlotSession API。
2. Dataset保持`(R,P,*data_dim)`与scalar`(R,P,1)`，Camera monitor每次publication为`(1,1,*frame)`且dtype不变；regular/sparse/serpentine/arbitrary/duplicate/P=1 point数据均可画，非grid sequence不Cartesian-expand，只有显式GridTopology可densify。R、P/grid logical dimensions与data axes都有coordinate/group/facet/sample/index/repeat-reducer明确归属；Histogram不隐式flatten，Curve/Image不聚合隐藏data axis，Rolling history不进入R/P/data_shape，生产代码不存在`MONITOR_HISTORY` Dataset role、`history_cycles`或history-in-P producer。adapter共享readonly source memory且不复制full validity/float image。
3. Cross/Area/Fit/Processor都携全部exact parents；Area→第二Figure、ROI→ROI→Fit与raw panel在同一SignalFront原子推进，siblings不拆、无关producer可独立前进，GUI tick不因plot暂缺fatal。Cross发布值、Area发布dtype/axes/validity保持的Dataset、Rolling多样本selection/Fit不丢source revision，Fit只发布参数；无hover与presentation sidecar。
4. Curve/Image/Histogram/Rolling/FacetGrid/PulseTimeline、Distribution自动双高斯/threshold、focused/all-facets Fit全部通过同一zlc_plot engine。Calibration、TaskConsole、DataFigure、FigureViewer、Edit与Pulse preview对同spec/size/DPR产生同style/data-box；Live Fit不停base，Edit Fit不冻结Monitor，Fit overlay直接在图中且不显示伪lag状态。
5. DeviceManager完成多种DeviceInstance的Add/Remove/Retype/New/Load/Save/Apply、domain分组与discovered/loaded分离；installation graph对missing/ambiguous/wrong-capability/cardinality/cycle启动即失败。active Run期间Apply不关闭/遗弃旧Run，idle Apply严格按validate→old SAFE/release→new connect→publish执行；连接失败不留下双owner或伪active graph，成功或恢复后public Experiment identity不变且旧runtime-bound窗口已退役。新增/删除device type只改leaf；DeviceRef按stable id钉住实例，runtime只有通用capability resolver，无flat backend form、具体Port中央表、中央concrete import或config digest。
6. 所有Logic Node均由一个generic host生成lifecycle/form/API/publish/persistence；新增/删除普通node只改leaf。Task takeover有header/progress/stage/Stop并禁止其它mutation，Measurement/Processor row-local，Occupancy不auto-open。Logic Edit与Plot Edit共享producer draft，Area可预填Camera ROI且Apply后继续采集。
7. 正式纵切全部通过：Camera N=3 atomic frames、Occupancy手动绑定、Calibration capture/calibrate/save_frames/reload/report、MOT Ready→live→FINAL、PulseScan、duration fidelity、release-recapture、FigureViewer/Edit/Grid batch Fit。Camera/MOT并行与真实冲突retire-then-start遵守§5.1；纯consumer不伪claim设备。
8. task/run输出位于用户project可见目录，只含可reload record、必要科学arrays与显式附件；普通SHA/fingerprint/tagged bytes/`$zlc-bytes`/size manifest/CAS/repository lease/软件内存预算在生产、测试和文档中为零。artifact写入不持hardware lease，partial目录不能load为成功。
9. PulseGUI Stop/hold/step通过窄replace-applied命令真实compile/upload/prepare/fire，稳定显示held point并缓存同revision；Remote/Virtual/Offline正式流程通过。PulseScan仍为AUTONOMOUS_STREAMED；真机E0按§6.3验证qCMOS路径与per-run reconcile。RTL/Tcl/XDC/bitstream/wire protocol diff为零。
10. GUI快轨和必要慢轨通过；resize不先stretch后跳，selector drag/zoom/pan连续且不锁死，普通字段编辑不做全局snapshot或重建widgets。1024/2048/2304方形uint8/uint16 live与Fit不形成H×W meshgrid或full float64副本，Qt callback p95<50ms且无>100ms Fit stall；结论有profile证据。
11. 所有公开Device/Logic Node至少通过discovery、typed form、construct、start/stop virtual smoke；被替代owner的测试已同cut改写或删除。每个cut有固定口径LOC/class/dataclass/enum、删除owner和真实consumer说明，整体生产代码显著净减，不存在单成员enum、单实例wrapper、零消费者DTO或空目录。
12. 全仓tracked文件change-impact审查完成；Workbench无具体Device/Logic Node/plot特判，public API无具体leaf imports，唯一架构文档、实现、教程与tests一致。`pulses/scan_test.json`未被读取、修改、stage或commit；临时审查台账在最终闭合后删除。

## 10. 收敛结论

最终架构不需要第二plot/data schema、异步workflow编排器、硬件重构、persistent quarantine、软件内存预算、普通CAS/SHA、per-revision presentation对象、为Fit预设的独立进程、device backend巨表或每leaf一套lifecycle。最关键的五个跨域语义authority是：zlc_data保存实验数据事实，SignalPlane保存因果，zlc_plot保存全部绘图交互，generic Logic host保存节点骨架，Experiment保存设备composition/admission；zlc_storage、zlc_pulse、zlc_frontend与Workbench仍分别拥有§2列出的窄职责，不能被这五项吞并。

C0–C6只是替换依赖顺序，不是可并存架构。每个cut必须以旧owner删除和真实产品证据结束；最终资格只由§9完整证据决定，单元测试通过不能替代。
