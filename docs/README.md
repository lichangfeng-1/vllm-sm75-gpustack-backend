# 文档索引（2026-10-27）

先读根 [README](../README.md)；需要动手时按下面顺序。

## 部署与运行

- [run-book](../gpustack/run-book-v1.md)：0→1 部署、粘贴块、排错、参数逐条说明。
- [backend card](../gpustack/backend-version-card-v1.md)：后端注册表单逐项。
- [compat matrix](../gpustack/compat-matrix-v1.md)：能力落点与读数边界。
- [release notes](../gpustack/release-notes-v2.3.0-v1.md)：GPUStack 2.3.0 升级与新特性。
- [configuration-v0.1.7](configuration-v0.1.7.md)：模型配置示例。
- [set-panel-autostart](../gpustack/set-panel-autostart-v1.sh)：面板入口与 env 一键写。

## 性能与调优

- [flash-next-20261027](tuning/flash-next-20261027.md)：当前 63.30 基线、负结果、硬件地板。
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
- [releases/](releases/)：历史发布说明与 manifest。
- [validation/](validation/)：按日期的验收记录。
- [benchlocal](benchlocal.md)、[lmcache](lmcache.md)：可选能力。

> 历史件保留原名与日期；当前验收线以根 README 的 2026-10-27 生产行为准。
