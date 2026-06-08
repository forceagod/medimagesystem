# DMP Trace 字段置信度打分标准

> 对应代码：`build_dmp_trace.py` → `compute_confidence()` (line 1012-1116)

---

## 评分维度

每个字段输出 **4 个评分 + 1 个方法标识**：

| 维度 | 含义 | 方向 |
|------|------|------|
| `extraction_accuracy` | 提取准确度 | 0-100，越高越好 |
| `completeness` | 内容完整度 | 0-100，越高越好 |
| `hallucination_risk` | 幻觉风险 | 0-100，**越低越好** |
| `overall_confidence` | 综合置信度 | 0-100，越高越好 |
| `extraction_method` | 提取方法标识 | 见下表 |

---

## 提取方法基础分

| extraction_method | 触发条件 | accuracy | completeness | hallucination_risk |
|---|---|---|---|---|
| `dm_literal` | DM日志精确匹配，key 完全一致 | **95** | 90 | 5 |
| `dm_lookup` | DM日志字段直接命中 | **90** | 90 | 5 |
| `protocol_literal` | 方案正文直接匹配 | **90** | 90 | 5 |
| `table_row` | 方案表格行直接命中 | **90** | 90 | 5 |
| `combined_fields` | 多个字段组合拼装（如页眉） | **85** | 80 | 5 |
| `dm_fuzzy` | DM日志近似匹配，证据含"近似"标记 | **70** | 80 | 5 |
| `derived` | 由其他字段推导计算得出 | **70** | 65 | 5 |
| `protocol_regex` | 方案正则表达式提取 | **65** | 60 | 5 |
| `protocol_search` | 方案全文关键词搜索 | **55** | 60 | **25** |
| `protocol_keyword` | 方案关键词匹配 | **50** | 60 | **25** |
| `user_confirm` | 弹窗由用户确认输入 | **40** | 50 | **50** |
| `none` | status 为 not_applicable / condition_pending | 0 | 0 | 0 |

---

## 完整度（completeness）修正规则

基础分由 `extraction_method` 决定后，根据证据质量下调：

| 触发条件 | 上限 |
|---|---|
| 证据含 "截取" 或 "fragment" | ≤ 50 |
| 证据含 "多行" 或 "multiple" | ≤ 70 |
| status = `uncertain` 或 `conflict` | 固定 30 |
| status = `missing` 或 `manual_confirm` | 固定 10 |

---

## 幻觉风险（hallucination_risk）修正规则

基础分由 `extraction_method` 决定后，根据以下条件上调（越高越危险）：

| 触发条件 | 最低值 |
|---|---|
| 有证据（evidence 非空） | ≤ 5，protocol_keyword/search 除外 |
| protocol_keyword / protocol_search（关键词匹配不精确） | ≥ 25 |
| status = filled 但无 evidence | ≥ 60 |
| status = uncertain / conflict / manual_confirm | ≥ 40 |
| status = missing | ≥ 70 |
| source_used 含 "AI" 或 "推断" | ≥ 60 |
| source_used 含 "用户确认" | ≥ 50 |

---

## 综合置信度（overall_confidence）计算公式

```
overall_confidence = accuracy × 0.35 + completeness × 0.35 + (100 - hallucination_risk) × 0.30
```

权重分配：准确度 35% + 完整度 35% + 反幻觉 30%  
结果四舍五入取整。

---

## extraction_method 判定流程

在 `process_row()` 中根据来源逐字段判定：

```
source_type = "DM日志"
  └─ 精确匹配（STRICT_IDENTIFIER_ITEMS）→ dm_literal
  └─ 近似匹配（证据含"近似"）→ dm_fuzzy
  └─ 直接命中 → dm_lookup

source_type = "方案"
  └─ 正则提取（protocol_lookup 命中）→ protocol_regex
  └─ 正文匹配 → protocol_literal 或 table_row

source_type = "DM日志/方案" 或 "方案/DM日志"
  └─ 两源冲突 → 取优先源的方法

特殊项：
  └─ 页眉/签署页配置 → combined_fields
  └─ 适用条件不满足 → none (overall_confidence = 0)
```

---

## 示例

### 高置信度（DM日志直接命中）

```json
{
  "extraction_method": "dm_literal",
  "extraction_accuracy": 95,
  "completeness": 90,
  "hallucination_risk": 5,
  "overall_confidence": 93
}
```

### 中置信度（方案正则提取，如本次其他终点）

```json
{
  "extraction_method": "protocol_regex",
  "extraction_accuracy": 65,
  "completeness": 60,
  "hallucination_risk": 5,
  "overall_confidence": 72
}
```
计算：`65×0.35 + 60×0.35 + 95×0.30 = 22.75 + 21 + 28.5 = 72.25 ≈ 72`

### 低置信度（缺失字段）

```json
{
  "extraction_method": "none",
  "extraction_accuracy": 0,
  "completeness": 0,
  "hallucination_risk": 0,
  "overall_confidence": 0,
  "scoring_note": "不适用"
}
```
