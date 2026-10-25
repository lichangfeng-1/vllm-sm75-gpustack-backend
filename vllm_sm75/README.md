# vllm_sm75 — SM75 (Turing) 可选优化包

**状态：骨架。当前没有任何已验证的项，`AUTO_ENABLED` 故意为空。**

## 门控顺序（严格短路）

```
compute capability == (7, 5)
  → VLLM_SM75_OPT ∈ {auto, 0, 1}
    → 逐项 VLLM_SM75_<ITEM> ∈ {auto, 0, 1}
      → 启动参数（优先级最高）
```

- `VLLM_SM75_OPT=0` 时，除本模块外**不导入任何子模块**，行为必须与基线镜像逐字节等价。
- CC 判定是**延迟**的：`install()` 在 `import vllm` 期间触发，那时 CUDA 可能未初始化，
  所以真正的能力检查放在首次 `enabled()` / hook 调用时。
- 每项默认 **off**；只有通过 L2（数值）、L3（指纹/PPL）、L4（同尺 A/B ≥3%）的项
  才能进入 `registry.AUTO_ENABLED`。

## 加一个优化项

1. 新建 `vllm_sm75/<name>.py`，暴露 `install()`（注册 hook / 替换 dispatch，不要改上游文件）；
2. 在 `registry.ITEMS` 登记：`{"module": ..., "default": "off", "status": "candidate", "l4_gain_pct": null}`；
3. 过 L2/L3/L4 后，把 `status` 改成 `passed`、填 `l4_gain_pct`，再决定是否加入 `AUTO_ENABLED`；
4. 更新 `manifest.json` 的 `items` / `verified_items`。

## 已验证但不在本包里的东西

`VLLM_FIREFLY_AR=0`（关 Firefly FP8 all-reduce）是**基线配置变更**，不是本包的开关：
它把 32K prefill 从 11.82 s 降到 10.46 s（+11.5%），decode 不受影响。
详见 `GPUStack使用SM75\v10-turing-借鉴-20261024\分析-Flash确定性优化-20261026.md`。

## 设计约束

- 不追上游：所有改动进我们自己的发行镜像，不以"能 upstream"为前提；
- 不改上游文件主体；新增能力只放在本包；
- 全关等价基线；每项独立开关、独立回退。
