# Chinese DMP Generation Rules

The authoritative rulebook for generating a Chinese DMP. Every rule below is binding.

## Contents

1. [Source Priority](#1-source-priority)
2. [Base Template Selection](#2-base-template-selection)
3. [Strict Project Identifiers](#3-strict-project-identifiers)
4. [Protocol Semantic Matching & Section 3](#4-protocol-semantic-matching--section-3)
5. [Version Revision History](#5-version-revision-history)
6. [Multi-Round DM Log (多轮对话)](#6-multi-round-dm-log-多轮对话)
7. [Checklist Statuses & Trace](#7-checklist-statuses--trace)
8. [Missing-Info Questions](#8-missing-info-questions)
9. [Template-Faithful Drafting](#9-template-faithful-drafting)
10. [In-Template Block Selection](#10-in-template-block-selection)
11. [Protected Table-Like Sections](#11-protected-table-like-sections)
12. [Semantic Review of High-Risk Fields](#12-semantic-review-of-high-risk-fields)
13. [Few-Shot Format Constraints](#13-few-shot-format-constraints)
14. [Final QA](#14-final-qa)

## 1. Source Priority

Use `assets/DMP非固定内容清单.xlsx` as the primary source of truth for every non-fixed item unless the user provides a newer checklist. Use the protocol as the main source for study-level facts and the DM log as the supplementary/project-reality source.

Read the DM log before the protocol for base Word template selection. Use the protocol for study facts after the template has been selected.

| Checklist `来源类型` | Required handling |
| --- | --- |
| `方案` | Extract from the clinical trial protocol. Preserve protocol names, numbers, versions, objectives, endpoints, sample sizes, and analysis populations as written. If only an inferred value is available, mark `uncertain` internally and include it in the final sequential clarification prompts when still unresolved. |
| `DM日志` | Prefer the DM log for project-reality fields such as dates, systems, vendors, deliverables, service scope, QC level, and decisions confirmed outside the protocol. Exact DM-log keys matching `非固定内容` are strong evidence. |
| `暂不处理` | Do not auto-fill unless the user confirms. Keep template default/fixed content unchanged and list the item as `manual_confirm` or `not_processed`. |
| blank/null | Treat as a governance or QC row. Use it to check trace quality, conflicts, or missing values; do not write it into the DMP as content. |

### Applicability Conditions

Checklist rows may include `适用条件` in `字段名=期望值` format, such as `是否使用随机系统=是` or `项目数据采集模式：EDC / PDC=EDC`.

- Blank `适用条件` means the row is always applicable.
- Multiple conditions may be separated with `;` or `；`; every condition must be satisfied before the row is applicable.
- Evaluate `适用条件` before normal extraction for that row.
- If the condition is met, continue with the row's normal source and missing-value handling.
- If the condition is clearly not met, mark the row `not_applicable`; do not generate a question and do not list it as missing in the DMP generation report.
- If the condition field itself is missing or unclear, mark the child row `condition_pending`; do not ask for the child value yet. The parent condition row should be asked/confirmed first.

When protocol and DM log disagree, do not choose silently. Record both source excerpts and include the item in the final sequential clarification prompts when still unresolved.

### Protocol/DM Log Cross-Checks

Before drafting, perform a lightweight semantic cross-check between the protocol and the DM log. This is a quality reminder step only. It must not automatically overwrite DM-log values, change the selected base template, delete template content, or invent missing DM-log decisions.

Flag the following situations for user confirmation when the evidence is clear:

- **Project type mismatch**: the protocol appears to describe a drug project, but the DM log says `项目类型：药物 / 器械 = 器械项目`; or the protocol appears to describe a device project, but the DM log says `药物项目`.
- **Randomization signal vs registration-only DM log**: the protocol clearly mentions randomization, random assignment, central randomization, IWRS, trial/control groups, parallel control, or two or more arms, but the DM log says `是否使用随机系统 = 否` or selects the registration-system template.
- **Single-arm signal vs randomization-system DM log**: the protocol clearly describes a single-arm, single-group, target-value, or no-control design, but the DM log says `是否使用随机系统 = 是`.

These rules are reminders, not deterministic template-selection rules. Multi-arm or controlled studies are often randomized, and single-arm studies are often registration-only, but exceptions exist. If a cross-check is triggered and remains unresolved at the final clarification gate, ask the user to confirm which DM-log value is correct as a sequential prompt. Include the protocol evidence and DM-log value in the prompt only. Do not require these lightweight cross-check reminders to be written into `dmp_questions.md`, and do not write them into the DMP document or `DMP生成报告.md` unless the user explicitly asks for an audit record.

### DM Log Internal Consistency Checks

Before drafting, perform a lightweight consistency check inside the DM log itself. For multi-round DM logs, check only the latest entry for current project-state fields; older entries are used for version history and should not trigger current-state consistency reminders.

These checks are conversation-only reminders. Include unresolved checks in the final sequential clarification prompts. Do not require them to be written into `dmp_questions.md`, do not write them into the DMP document or `DMP生成报告.md`, and do not automatically change or clear any DM-log value unless the user confirms.

Flag the following situations for user confirmation when the parent value and child value are both clear:

- `是否使用随机系统 = 否`, but `随机系统供应商/系统类型` is non-empty. Ask whether the random-system decision should be changed to `是` or the random-system vendor should be cleared.
- `项目数据采集模式：EDC / PDC = PDC`, but `EDC系统供应商/系统类型` is non-empty. Ask whether the data-capture mode should be changed to `EDC` or the EDC-system vendor should be cleared.
- `是否涉及外部数据 = 否`, but `设计的外部数据类型` is non-empty. Ask whether the external-data decision should be changed to `是` or the external-data type field should be cleared.
- `是否有阶段性分析/中期分析 = 否`, but `阶段性分析目的和阶段要求` is non-empty. Ask whether the stage-analysis decision should be changed to `是` or the stage-analysis details should be cleared.

The inverse cases, such as `是否使用随机系统 = 是` with an empty random-system vendor, are handled by normal checklist applicability and missing-value logic. Do not duplicate them as internal-consistency reminders.

Never encode example-project facts in the skill or scripts. A sample protocol may demonstrate behavior, but names, identifiers, sponsors, indications, vendors, URLs, endpoints, dates, and decisions must always come from the active input files and checklist trace.

## 2. Base Template Selection

Select exactly one Word template from the current DM log before reading or applying template content:

| DM log decision | Action |
| --- | --- |
| `是否使用随机系统 = 是` and `是否使用登记系统 = 是` | **Stop and ask the user to clarify.** A project will not use both a randomization system and a registration system simultaneously; one of the two DM log fields is incorrect. |
| `是否使用随机系统 = 是` | `assets/DMP-随机系统.docx` |
| `是否使用随机系统 != 是` and `是否使用登记系统 = 是` | `assets/DMP-登记系统.docx` |
| both `是否使用随机系统` and `是否使用登记系统` are `否` | `assets/DMP-无随机无登记.docx` |

Do not infer the base template from protocol text unless the DM log is missing or unclear and the user confirms. If the DM log uses nested fields, flattened keys, or equivalent field names, use semantic key matching; if the decision still cannot be made, stop and ask.

After this base template is selected, all extraction, replacement, and drafting must use only that selected Word file. Do not mix content from other base templates.

## 3. Strict Project Identifiers

The following fields must be extracted from the current protocol and/or DM log, never invented:

- `方案名称` / `临床试验方案名称`
- `方案编号`
- `申办方名称` / `申办者名称`
- `数据管理单位名称`

Look in the protocol first and also check the DM log for confirmed project metadata. If both sources provide incompatible values, mark `conflict` internally and include the field in the final sequential clarification prompts. If neither source provides the field, leave it unresolved and include it in the final sequential clarification prompts. Do not fill generic sponsor, protocol, or data-management-unit placeholders silently.

## 4. Protocol Semantic Matching & Section 3

Checklist source locations are guidance, not exact heading requirements. If a row says to use `试验摘要`, equivalent protocol areas such as `方案摘要`, `研究摘要`, `临床试验摘要`, or a table containing the same summary fields may be used. The extracted value must still be clearly present in the protocol evidence.

### Section 3 试验概述

For Section `3 试验概述`, prefer the protocol summary table/section and paste the protocol wording as directly as possible for study name, design, purpose, sample size, endpoints, and analysis population. If the summary is incomplete, search equivalent sections such as study design, endpoints, and analysis dataset chapters before asking.

When the protocol is `.docx`, extraction should be structure-first: use Word tables, row labels, cell paragraph order, and adjacent semantic blocks. Preserve paragraph/newline order and do not split medical content by punctuation such as `；`, `,`, `。`, or by generated separators such as `|`. For example, if the main endpoint has a following definition paragraph in the same summary cell, keep that definition with the main endpoint until the next endpoint category begins.

When the protocol is `.pdf` or plain text, use semantic matching against the extracted text but keep the same conservative standard. If the source structure does not make the full endpoint/objective/design block clear, mark it `uncertain` or `missing` internally and include it in the final sequential clarification prompts when still unresolved.

### 其他终点

For `其他终点`, collect all non-primary endpoint content that is clearly present in the protocol summary or equivalent overview area. This includes secondary endpoints, safety endpoints, and exploratory endpoints, whether they appear in one summary cell or in separate adjacent rows such as `安全性指标` or `探索性终点`. Do not stop after the first secondary endpoint block if the protocol summary has additional safety/exploratory rows. Preserve the row label when it is needed to keep the endpoint type clear.

### 研究设计类型为xxx

Replace every `研究设计类型为xxx` occurrence with the study design wording extracted from the protocol summary when clearly available. Do not standardize or rewrite the design wording if the protocol text is clear.

## 5. Version Revision History

Generate the DMP version revision table from DM-log version records:

- one DM-log version record -> one table row
- multiple DM-log version records -> one table row per record
- incomplete version fields -> include the missing fields in the final sequential clarification prompts

Preserve the selected template's revision table structure. Do not merge version records and do not invent version numbers, dates, authors, or revision content.

Sort version records from oldest to newest by version date, then by version number, so the revision history reads chronologically downward.

## 6. Multi-Round DM Log (多轮对话)

When the DM log JSON contains an array of multiple entries (e.g. `[{...}, {...}, {...}]`):

- **Latest entry wins for non-version fields**: Use the last entry in the array for all project-state fields such as `项目类型`, `EDC系统供应商`, `随机系统供应商`, `是否涉及外部数据`, `是否有阶段性分析`, `项目质量控制等级`, etc. The last entry represents the most current project reality.
- **All entries contribute to version history**: Every entry that contains version fields (`DMP版本号`, `DMP版本日期`, `撰写者/修订者`, `版本修订记录`) becomes a row in the version revision history table, sorted from oldest to newest.
- **DMP version metadata**: The latest entry's `DMP版本号` and `DMP版本日期` are used as the current DMP version for cover pages, headers, and signature pages.
- If the DM log is a single entry (not an array), behavior is unchanged from the single-round case.

Do not infer which entry to use from protocol text. The DM log array order is authoritative: first entry = oldest, last entry = newest.

### Signature Page Signers

Signature-page signer fields should be flat fields on the current DM log entry:

- `撰写人` (optional; if absent, fall back to `撰写者/修订者`)
- `数据管理单位审核人`
- `申办者审核人`
- `CRO审核人`
- `统计分析单位审核人`

Extract these fields into `trace.metadata.signature_signers`. For multi-entry DM logs, signer fields are current project-state fields, so use the latest entry that provides these flat signer keys. Legacy nested `签署页签署人` objects may be accepted for backward compatibility, but new DM logs should not use that nested shape.

The normalized metadata must use this shape:

```json
{
  "writer": "optional writer name",
  "reviewers": {
    "key 1": {"role": "数据管理单位审核人", "name": "..."},
    "key 2": {"role": "申办者审核人", "name": "..."},
    "key 3": {"role": "CRO审核人", "name": "..."},
    "key 4": {"role": "统计分析单位审核人", "name": "..."}
  },
  "raw": {"original DM-log signer field": "original value"}
}
```

Checklist row `签署页配置` represents the signature-page writer and maps directly to DM-log `撰写者/修订者`; it should not trigger a separate user question when `撰写者/修订者` is present. The selected DOCX templates use `审核人：key 1` through `审核人：key 4` on the signature pages. Fill those reviewer placeholders from the normalized metadata in key order. For writer placeholders, use flat `撰写人` when present; otherwise fall back to the confirmed `撰写者/修订者` value. Missing reviewer names should remain visible as apply-stage warnings, not silent blanks; include those missing signer names in the final sequential clarification prompts before final delivery.

## 7. Checklist Statuses & Trace

### Statuses

Use these statuses in the trace:

- `filled`: value is supported by the specified source and can be used.
- `uncertain`: a plausible value exists but evidence is not strong enough.
- `missing`: required value is absent from the specified source.
- `conflict`: protocol and DM log provide incompatible values.
- `manual_confirm`: checklist says the item needs user confirmation or is `暂不处理`.
- `not_processed`: item should not be automatically handled in this first version.
- `not_applicable`: row has an `适用条件`, and the condition is clearly not met.
- `condition_pending`: row has an `适用条件`, but the condition field is missing or unclear; resolve the parent condition before asking the child field.
- `qc_rule`: row is a quality/control rule, not a DMP field.

Only `filled` values should be applied automatically. User answers may be written back into the trace as `filled` with `source_used: user_confirmation`.

### Trace Requirement

Keep a trace for each checklist row with: section, item, source type, value, evidence, status, and question if unresolved. This trace is the audit record for every non-fixed change in the final DMP.

## 8. Missing-Info Questions

`dmp_questions.md` is the initial confirmation draft from the trace-build pass. It is not required to contain every question later raised by semantic review, few-shot formatting, protocol/DM-log cross-checks, DM-log internal consistency checks, apply-stage placeholder checks, or manual agent review.

Collect confirmation candidates during trace review, semantic review, few-shot formatting, protocol/DM-log cross-checks, DM-log internal consistency checks, and manual review, but do not ask them immediately just because they appear in `dmp_questions.md`.

At the final clarification gate, after semantic review, few-shot formatting, manual review, template apply, and apply-stage placeholder retry/checks have run, filter the candidate set to only items that still remain unresolved. Do not ask about fields that became `filled`, `not_applicable`, user-confirmed, or were successfully filled by apply-stage second-pass placeholder retry.

Ask the user one unresolved item at a time in the conversation. Do not send one combined numbered list. The actual prompt may use the user's language, but each individual prompt should follow this structure:

```text
确认 1/5: Confirm "field or decision": current evidence is ...; please reply with the value or handling decision to use.
```

Do not expose internal trace labels such as `missing`, `uncertain`, `conflict`, `manual_confirm`, or `not_processed` in the user-facing question text. Use those labels only inside the trace/reporting workflow.

Each question should still be precise enough for the user to answer:

- DMP section/application scope when relevant
- Non-fixed content item or DM-log decision being confirmed
- Expected value or decision
- Current evidence, if any
- What the user should reply with

Avoid asking vague questions like "please confirm project info." Ask for the exact field or decision. The user answers the current prompt; then the agent continues to the next still-unresolved prompt if one remains.

After each user confirmation, update the trace and, when the answer corrects or supplies a DM-log field, update the latest DM-log JSON entry. If the answer affects generated content, rerun the downstream apply/report steps before final delivery. Do not ask stale questions that are already resolved by semantic review, few-shot formatting, agent review, user confirmation, or apply-stage second-pass placeholder retry. The final `DMP生成报告.md` should list only unresolved items that remain after those confirmations.

## 9. Template-Faithful Drafting

### Fixed Content

Copy fixed content exactly from the Word template. Do not rewrite, polish, summarize, translate, reorder, renumber, or add sections. Always start from the Word template by copying it — do not generate a DMP from a blank document.

### Safe Automatic Fills

These may be applied without user confirmation when evidence is sufficient:

- Cover/version labels when a value is confirmed.
- Trial overview table rows whose first cell exactly matches a checklist field.
- Specific placeholder words such as `XXXX` or `xxx` only when the surrounding sentence clearly matches the checklist item.
- Body, cover, signature-page, header, and footer placeholders when the field is confirmed, including synonymous placeholders such as `请输入申办者`, `请输入申办者名称`, `请输入临床监查方`, and `请输入临床监察方`.
- Signature-page writer placeholders when the DM log provides flat `撰写人` or `撰写者/修订者`; prefer `撰写人` and normalize duplicate labels such as `撰写人：撰写人：姓名`.
- Signature-page reviewer placeholders `审核人：key 1` through `审核人：key 4` when `trace.metadata.signature_signers.reviewers` provides confirmed names.
- Protocol version number/date should come from the protocol body first. If Word extraction produces a visibly incomplete value but the file name provides the missing version/date, the file name may be used as lower-priority evidence and recorded in the trace.

After the normal fill pass, scan only the cover and signature-page regions for unresolved `请输入...` placeholders. A second-pass lookup may replace them only when a confirmed trace value clearly maps to the placeholder. Any remaining cover/signature placeholders are stderr diagnostics; they are not automatically added to `dmp_questions.md` or `DMP生成报告.md` unless the user asks for an audit record.

### Unsafe Automatic Edits

These must never be done automatically:

- Rewriting fixed paragraphs.
- Removing template sections without a checklist rule and confirmed evidence.
- Reconstructing tables manually.
- Applying inferred values without asking.
- Using the example protocol as a format requirement.

## 10. In-Template Block Selection

For checklist rows with `判断粒度` such as `统一模板选择`, `适用性判断`, or `统一联动判断`:

1. Confirm the governing decision once in the trace.
2. Locate the relevant existing `/模板.../` block in the Word template.
3. Keep the selected block's wording as-is except for governed placeholders.
4. Remove non-selected blocks only outside protected table-like sections and only when the decision is `filled`.
5. If no existing block fits, include the decision in the final sequential clarification prompts. Do not draft a new block.

Resolve `/模板1/`, `/模版2/`, `/*模版...*/`, and similar marker labels across the whole draft. Selected marker labels must not appear in the final DMP.

### Body-Level END Markers

Body-level selectable template blocks must have explicit standalone boundaries:

- The start marker may keep any existing format, such as `模板2：适用于器械项目，`, `/*模版1*/EDC项目适用`, or `/模版1/适用于PDC项目`.
- The END marker must be a standalone paragraph whose text is the full start marker plus `END`, for example `模板2：适用于器械项目，END`. A single space before `END` is tolerated, for example `/*模版1*/（适用于PDC项目） END`.
- When a body block is selected, delete only its start marker paragraph and END marker paragraph; keep the actual template content between them.
- When a body block is not selected, delete the start marker, all content through the matching END marker, and the END marker.
- If a body start marker is present but the matching END marker is missing, stop and fix the Word template rather than guessing the block boundary.
- Table-cell template choices are not body-level blocks. Do not add END markers inside table cells; keep using the table-specific selection logic for those cells.

Template-selection code may map generic checklist values to existing template markers, for example `EDC` -> EDC blocks or a current system string containing `太美` and `V6` -> the bundled `太美系统V6` block. Do not map by project name, protocol number, sponsor, disease, or example-file identity.

If a selection row is `conflict`, keep the conflict in the questions/report. If a draft must still be produced, any provisional selection must be based on generic source evidence and clearly remain pending confirmation; never hide the conflict.

### Other-System Detail Fields

Keep these parent fields as the template-selection keys:

- `EDC系统供应商/系统类型`
- `随机系统供应商/系统类型`

When `EDC系统供应商/系统类型` selects `/模板7/其他系统`, fill only the selected 7.3 other-system block from these flat DM-log keys:

- `EDC其他系统供应商名称`
- `EDC其他系统名称`
- `EDC其他系统维护负责方`
- `EDC其他系统版本号`
- `EDC其他系统服务器地址`

The EDC other-system block keeps the template's fixed wording that database build is handled by `蓝气球（北京）医学研究有限公司`; do not add a separate DM-log key for that fixed clause unless the template itself changes.

When `随机系统供应商/系统类型` selects `/模板4/其他随机系统`, fill only the selected 8.1 other-random-system block from these flat DM-log keys:

- `随机其他系统供应商名称`
- `随机其他系统搭建负责方`
- `随机其他系统维护负责方`
- `随机其他系统版本号`
- `随机其他系统服务器地址`

These child rows are applicable only when the parent selection chooses an other-system block. Missing child values remain unresolved confirmation candidates; do not infer company names, versions, or server addresses from the parent system string.

## 11. Protected Table-Like Sections

The first version must preserve all existing items in these sections:

- Section 9
- Section 15.2
- Section 26.1
- Section 27.1
- Section 27.2
- Section 27.3
- Section 29

For these sections, do not attempt to infer checkbox status, selected items, deletion, or applicability. Keep all current template rows/items exactly so the data manager can adjust them later.

It is acceptable to replace inline `/模板1/… /模板2/…` text inside a protected table cell with the selected wording when the row itself is preserved and the checklist provides a confirmed generic decision. Do not remove protected rows/items automatically.

## 12. Semantic Review of High-Risk Fields

The trace extractors use pure rules (table structure, regex, keyword matching) without semantic understanding. Fields such as `样本量`, `研究设计`, `主要有效性终点`, `其他终点`, and `统计分析人群` are prone to extraction errors — e.g., picking an intermediate sample size instead of the final total. These fields must undergo semantic review before drafting.

Common semantic checks:

- **样本量**: Is this the FINAL total sample size (including dropout), not an intermediate calculation? Look for "最终样本量", "总样本量", "所需样本量", "考虑脱落率后" in surrounding context. Take the largest number when multiple are present.
- **研究设计**: Does the value capture the full study design (phase, arms, blinding, control type)? Check the protocol summary row for completeness.
- **主要有效性终点**: Is the primary endpoint complete and correctly identified (not a secondary or safety endpoint)?
- **其他终点**: Are ALL secondary, safety, and exploratory endpoints captured? Check for missing endpoint categories like `安全性指标` or `探索性终点`.
- **统计分析人群**: Are all analysis populations included (FAS, PPS, SS)?

Review decisions: `accept` (value is correct), `correct` (value is wrong; provide `corrected_value` and `correction_reason`), or `flag` (unclear, needs user input).

Semantic review runs AFTER the initial trace build and BEFORE drafting. See the workflow in SKILL.md for the script commands.

## 13. Few-Shot Format Constraints

When a `fewshot.md` file is provided, apply few-shot format constraints to matching fields AFTER semantic review correction. The few-shot file defines per-field reference examples; reformat the corrected trace values to match the example style (conciseness, sentence template, placeholder substitution) before drafting.

Common format checks:

- **研究设计**: Is the value a single concise sentence (type + arms + blinding + center + control), or does it include sample size / evaluation / period / administration details that belong in other fields?
- **样本量**: Does the value follow the template sentence pattern shown in the few-shot example, with actual numbers replacing placeholders?

Format decisions: `accept` (already matches example format), `reformat` (rewrite to match few-shot style; provide `formatted_value` and `format_reason`), or `flag` (unclear, needs user input).

Few-shot formatting runs AFTER semantic review so values are already corrected before reformatting. See the workflow in SKILL.md for the script commands.

## 14. Final QA

Before delivery, verify:

- The DMP was produced from the template `.docx` (a copy, not the original).
- Fixed sections were not rewritten, polished, summarized, translated, reordered, renumbered, or added to.
- Section numbering and heading hierarchy are unchanged unless the user explicitly required a confirmed template-block removal.
- Table count and protected table contents remain intact.
- Each non-fixed modification has a trace entry with evidence or user confirmation.
- No unresolved required fields remain silently blank.
- Protocol and DM log conflicts are listed for confirmation, not auto-overwritten.
- Missing/conflicting items are either answered by the user or clearly listed as pending.
- Apply-stage cover/signature placeholder warnings were reviewed before delivery.

### Final Response AI/Few-Shot/Targeted-Replacement Disclosure

After generating the draft, the agent's final response must include a compact user-facing disclosure for any trace item whose final value was produced or changed by AI semantic review or few-shot formatting, plus the selected Section 8.1 `研究设计类型为xxx` targeted replacement when present. This disclosure is part of the agent response only; do not generate an extra Markdown file and do not print an AI summary from the script.

Detect these items from the corrected `dmp_trace.json`:

- `source_used` contains `LLM语义审核` or evidence contains `LLM语义审核修正`
- `source_used` contains `few-shot格式化` or evidence contains `few-shot格式化`

Also inspect the generated draft or `DMP生成报告.md`. If the apply step records `定向句子 \`研究设计类型为xxx\` <- 研究设计`, include the final selected 8.1 design-type wording as `8.1 研究设计类型：...`. Use the actual wording in the generated 8.1 sentence, not a newly summarized value.

For each detected item, list the field and final value in `字段：内容` format. Tell the user to manually confirm accuracy, but do not ask for a reply and do not stop the workflow:

```text
以下字段经过 AI 语义审核、few-shot 格式约束或模板定向替换，请人工确认准确性（无需回复）：
- 样本量：...
- 研究设计：...
- 8.1 研究设计类型：...
```

If no fields were touched by semantic review, few-shot formatting, or the 8.1 targeted replacement, omit this disclosure or state briefly that there were none.
