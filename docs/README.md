# 文档索引（2026-11-16）

先读根 [README](../README.md)；需要动手时按下面顺序。

## 部署与运行

- [run-book](../gpustack/run-book-v1.md)：0→1 部署、粘贴块、排错、参数逐条说明。
- [backend card](../gpustack/backend-version-card-v1.md)：后端注册表单逐项（含 2026-10-29 final-v1 / SQL 提示词审计增补）。
- [compat matrix](../gpustack/compat-matrix-v1.md)：能力落点与读数边界。
- [SQL 判分扩展 v20261029](../gpustack/layers/panel/extensions/sql/UPSTREAM-v20261029.md)：finalanswer 判分、财年格式、工具结果截断。
- [测速扩展 v20261028](../gpustack/layers/panel-responsive/public/UPSTREAM-v20261028.md)：配额安全与上传前缀。
- [release notes v2.3.1](../gpustack/release-notes-v2.3.1-v1.md)：GPUStack v2.3.1 升级、新特性门槛与未复核门禁（现役）。
- [release notes v2.3.0](../gpustack/release-notes-v2.3.0-v1.md)：GPUStack 2.3.0 升级与新特性（2026-10-23 历史件）。
- [configuration-v0.1.7](configuration-v0.1.7.md)：模型配置示例。
- [set-panel-autostart](../gpustack/set-panel-autostart-v1.sh)：面板入口与 env 一键写。

## 性能与调优

- [swift15-prod-20261116](tuning/swift15-prod-20261116.md)：**现役基线**——Swift-1.5 行 48（decode 均 66.68）、现读运行时值、显存口径 15929 MiB/卡、四组参数判定、空间边界与已判死清单。
- [flash-next-20261027](tuning/flash-next-20261027.md)：2026-10-27 Intel opt15 历史基线（63.30，已被 Swift 行 48 取代）、负结果、硬件地板。
- [compat-turing032-20261116](tuning/compat-turing032-20261116.md)：Turing 0.32 兼容件层（默认关）、**0.32 整栈非 MTP 净负（平均输出 59.47）故只吸收兼容件不整栈迁**、容量档 block400（只动容量不动速度）、MTP2 实测与用户裁定不做。
- [flash-next-performance](flash-next-performance.md)：2026-10-04 Ultra/面板历史读数。
- [flash-next-tp8](flash-next-tp8.md)：TP8 使用说明与历史配置。
- [request-timing](request-timing.md)：请求时序。
- [sleep-and-cache](sleep-and-cache.md)：休眠与缓存。

## 设计与调研

- [roofline-audit-20261027](design/roofline-audit-20261027.md)
- [hardware-utilization-20261027](design/hardware-utilization-20261027.md)
- [external-techniques-20261027](design/external-techniques-20261027.md)
- [next-projects-20261027](design/next-projects-20261027.md)
- [8bit-collective-20261027](design/8bit-collective-20261027.md)

## 发布与验证

- [v0.1.7-beta2 release note](releases/v0.1.7-beta2.md)：当前后端版本线。
- [v0.1.7-beta2 compat1](releases/v0.1.7-beta2-compat1.md)：Turing 0.32 兼容件层（默认关，未注册）。
- [releases/](releases/)：历史发布说明与 manifest。
- [validation/](validation/)：按日期的验收记录。
- [benchlocal](benchlocal.md)、[lmcache](lmcache.md)：可选能力。

> 历史件保留原名与日期；**当前验收线以根 README 的 2026-11-16 现役生产行为准（Swift-1.5 行 48，面板三档 decode 均 66.68）**。2026-10-27 的 63.30 是 Intel opt15 历史基线，保留在案并标明已被取代；不同窗口、不同权重、不同尺子的读数不得相减。
