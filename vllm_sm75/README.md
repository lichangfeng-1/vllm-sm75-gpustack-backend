# vllm_sm75 — SM75 (Turing) 可选优化包

**状态（2026-10-27）**：已转正 2 项（`gemv_m1`、`gdn_tc`，均在 `auto`），判负留档 4 项，
另有 2 个只测量的探针。全部默认关时可与基线镜像行为一致（`VLLM_SM75_OPT=0` 一行子模块都不加载）。

## 门控顺序（严格短路）

```
compute capability == (7, 5)
  → VLLM_SM75_OPT ∈ {auto, 0, 1}
    → 逐项 VLLM_SM75_<ITEM> ∈ {auto, 0, 1}
      → 启动参数（优先级最高）
```

- 非法 `VLLM_SM75_OPT` 值按 `0` 处理（失败关闭）。
- CC 判定是延迟的：`install()` 在 `import vllm` 期间触发，那时 CUDA 可能未初始化。
- 每项默认 **off**；只有通过 L2（数值）、L3（logprob）、L4（同尺 A/B ≥ **1%**、3–5 次中位）
  才能进入 `registry.AUTO_ENABLED`。**门控只在 `registry.py` 一处**，item 模块内不得再读自己的开关。

## 当前登记（`manifest.json` 是唯一真源）

| 项 | 状态 | 说明 |
|---|---|---|
| `gemv_m1` | verified / auto | M=1 逐形状 Triton GEMV；解码 64.32 → 65.24 tok/s（+1.4%） |
| `gdn_tc` | verified / auto | 张量核 GDN chunk 预填；**需 `--gdn-prefill-backend=triton`**；预填 +6.5/12.0/10.1%（8K/32K/126K） |
| `ar_prefill` | rejected | PCIe-IPC 服务预填 all-reduce，慢 15–22% |
| `ar_overlap` | rejected | RowParallelLinear 切块异步 AR，同镜像 A/B 无收益 |
| `qsa_slice` | rejected | QSA 选页切片 1024→4096 行，无收益 |
| `ar_q8_probe` | probe（永不 auto） | 只量化、不改通信，用于量 8-bit 集合通信的精度代价；按“不降精度”已作废 |
| `qsa_timing` | probe（永不 auto） | QSA 路径计时，实测全部 <2 ms |
| `prefill_tail_overlap` | implemented_pending_gate | 层尾重叠（o_proj→AR→HC→MLP 逐 token 分块，AR 走 side stream）；8K 合成 proxy +9.54%（上界），待 L2/L3 与 64K A/B |
| `hc_fuse` / `gdn_fuse` / `qsa_merge` | already_fused_upstream | 上游已融合，永不加载 |

## 加一个优化项

1. 新建 `vllm_sm75/<name>.py`，暴露 `install()`（注册 hook / 替换 dispatch，不改上游文件）。
2. 在 `manifest.json` 的 `items` 登记（`status`：`implemented_pending_gate` → `verified` / `rejected` / `probe`）。
3. 过 L2/L3/L4 后填 `l4_gain_pct`，再决定是否加入 `auto_enabled`；
   `verified` 但增益 <1% 的项不得进 `auto`（L1 测试会拦）。
4. 安装器失败要降级不要 500：`install()` 里的第三方编译/自检失败必须回退原生实现（`gdn_tc` 的 install 预检就是范例）。

## 已验证但不在本包里的东西

`VLLM_FIREFLY_AR=0`（关 Firefly FP8 all-reduce）是**基线配置变更**，不是本包开关：
32K prefill 11.82 → 10.46 s（+11.5%），decode 不受影响。

## 设计约束

- 只新增、不覆盖镜像里已有文件；全关等价基线；每项独立开关、独立回退。
- 不做精度换速度：量化/8-bit 集合通信/MTP 不在本包范围内。
- 判定与读数：`../docs/tuning/flash-next-20261027.md`；外部技术对照：`../docs/design/external-techniques-20261027.md`。
