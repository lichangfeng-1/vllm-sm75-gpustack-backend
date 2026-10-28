# 测速页 v20261028 本地补丁（配额安全 + 上传前缀）

- 基线：`speedtest-v20261021.html`（sha `fe139724…085048c`；上游 v3.5 原样 + 2 行响应式 CSS）。
- 本版：`speedtest-v20261028.html`，sha `cb05faa3cdc196c71da7903ac09fef36b8c25a04118211ba98efc10fdeb05bd1`，727,830 B。
- 部署名：`/opt/sm75-workbench/extensions/speedtest/LLM_Speed_Test_v3_Python_Backend.html`（final-v1 使用此件）。

## 改动

1. **历史记录配额安全**：`saveTestToHistory` 在 `QuotaExceededError` 时先递归剥离
   `promptText` / `outputText` / `reasoningContent` / `reasoningText`，仍失败则丢最旧记录重试，
   最后才提示“存储空间已满”；未勾选详情时补剥 `reasoningText`。
   根因是 v3.5 历史把 prompt/output + reasoningText + 6×8 对比数据全部塞进
   `localStorage` 的 `llmPerfTestHistory`，20 条上限挡不住体积。
2. **上传前缀**：页面自身 `BACKEND_BASE = ''`，`/api/upload` 会打到 console 根；
   `ultra/source/console/original-tests.mjs` 在返回 HTML 时注入
   `BACKEND_BASE = '/bench-app/speedtest'`，分享/用户 API 改走扩展前缀。
   页面本身不改这一行，避免破坏上游字节。

## 验证

- `grep -c 'QuotaExceededError'`、`stripHistoryDetail`、`reasoningText` 命中。
- final-v1 镜像内 canonical 页 sha 为 `cb05faa3…`；上传分享页请求路径为
  `/bench-app/speedtest/api/upload`。
- 旧版页仍可打开；回退不影响已存历史以外的数据。

## 回退

- 旧版备份：`…before-quota-20261028`（sha `fe139724…`）、`…before-uploadfix-20261028`（sha `b02e5475…`）。
- 恢复备份后刷新页面；若同时回退 console 注入，需换回旧镜像并重启实例。

## 与上游记录的关系

`gpustack/layers/panel/extensions/speedtest/UPSTREAM.md` 记录的是上游 v3.5 原样对齐（v20261021），
不是 final-v1 的实际入口；final-v1 使用本目录的 v20261028 本地补丁页。
