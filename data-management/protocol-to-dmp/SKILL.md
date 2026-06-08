---
name: protocol-to-dmp
description: Use when the user provides a clinical trial protocol, DM log, DMP rule/template document, or DMP non-fixed-content checklist and asks to generate, draft, update, or automate a Chinese Data Management Plan (DMP/数据管理计划) that must follow a fixed Word template and Excel rules.
---

# Protocol to Chinese DMP

Generate a Chinese DMP draft by selecting one governed Word template, filling only governed non-fixed content, and preserving fixed wording.

## Core Rules

`reference/chinese-dmp-generation.md` is the authoritative rulebook. The rules below are the operating summary; consult the referenced sections when a decision is relevant, ambiguous, or fails validation. Do not paste or summarize the rulebook in conversation unless the user asks.

## Output Discipline

Keep internal reasoning, trace review, semantic review, few-shot formatting, and field-by-field validation out of chat. Work through scripts and artifacts silently, and only show user-actionable output.

- All user-visible output must be in Chinese. This includes progress updates, blocking questions, the final response, and the AI-review disclosure.
- **Terminology**: `protocol` in this clinical-trial context always translates to **方案** (临床试验方案), never to 协议 (which means a network/communication protocol).
- Progress updates should be short milestone notes only（如：输入已解析、trace 已构建、初稿已生成、QA 完成），do not narrate internal steps.
- Do not print chain-of-thought, exhaustive evidence comparisons, raw trace JSON, review JSON, full script logs, or internal status labels. Store detailed evidence in `dmp_trace.json`, review files, and `DMP生成报告.md`.
- Ask only blocking questions that remain unresolved after automated review, one at a time, in plain Chinese. Do not ask for confirmation of fields that are already filled or not applicable.
- Final response should be concise: generated file paths, unresolved blockers if any, QA performed, and the required AI-review disclosure. Omit implementation narrative unless the user asks.
- For speed, prefer bundled scripts and targeted reads of the rulebook sections cited by the relevant Core Rule. Read the full rulebook only for rule/script updates, new template variants, or unusual conflicts.

1. **Template selection** — Read the DM log first. If both `是否使用随机系统` and `是否使用登记系统` are `是`, stop and ask the user to clarify (a project cannot use both). Otherwise: `是否使用随机系统 = 是` → `assets/DMP-随机系统.docx`, else `是否使用登记系统 = 是` → `assets/DMP-登记系统.docx`, else → `assets/DMP-无随机无登记.docx`. Never infer from protocol. (ref §2)

**Coupled-field rule**: `是否使用随机系统` and `是否使用登记系统` are interdependent template-selection fields. When the user corrects one of these fields during confirmation, you MUST re-read the updated DM log, re-evaluate the template conditions, and proactively ask whether the other coupled field also needs correction. Do NOT silently derive the final template from the corrected value alone — always verify the other field before proceeding.

2. **Source hierarchy & strict identifiers** — `assets/DMP非固定内容清单.xlsx` is the master for all non-fixed items. Protocol = primary source for study facts. DM log = project-reality source (systems, vendors, dates, deliverables). Extract `方案名称`/`临床试验方案名称`, `方案编号`, `申办方`/`申办者名称`, and `数据管理单位名称` from the current protocol and/or DM log. If sources conflict or neither provides the value, include the item in the final sequential clarification prompts; never fabricate. Never hard-code facts from example projects. (ref §1, §3)

3. **Fixed content & placeholders** — Copy fixed content exactly from the template. Do not rewrite, polish, summarize, translate, reorder, renumber, or add sections. Replace confirmed placeholders across body, tables, cover page, signature pages, headers, and footers. Support synonyms: `请输入申办者`/`请输入申办者名称`, `请输入临床监查方`/`请输入临床监察方`. For signature pages, checklist row `签署页配置` maps directly to DM-log `撰写者/修订者`; use flat DM-log `撰写人` only as an optional override. Reviewer placeholders use flat keys `数据管理单位审核人`, `申办者审核人`, `CRO审核人`, and `统计分析单位审核人` in this order for `审核人：key 1~4`. If any reviewer name is missing after the apply-stage retry/check, include it in the final sequential clarification prompts before final delivery. (ref §6, §9)

4. **Sequential final confirmation, don't guess** — If a required value or decision remains unresolved after all automated review/retry steps, ask the user one item at a time at the final confirmation gate. Do not ask as one combined list, and do not expose internal trace status labels such as `missing`, `conflict`, `uncertain`, `manual_confirm`, or `not_processed` to the user. (ref §8)

5. **Section 3 试验概述** — Extract from protocol summary (or equivalent: `方案摘要`, `研究摘要`, `临床试验摘要`). Preserve protocol wording and paragraph/newline order. For Word protocols, extract structure-first from tables, cell paragraphs, and adjacent semantic blocks. Include all non-primary endpoint blocks (secondary, safety, exploratory) — do not stop after the first secondary block. Replace every `研究设计类型为xxx` with the protocol's study-design wording. For PDF/plain-text, use the same conservative standard; include unclear structure in the final sequential clarification prompts. (ref §4)

6. **Version history** — One row per DM-log version record, sorted oldest to newest by version date then version number. Never merge or invent. For multi-round DM logs: latest entry governs non-version fields (systems, service scope, QC level); all entries contribute to the revision history in chronological order. DM log array order is authoritative: first = oldest, last = newest. (ref §5-6)

7. **Protected sections** — Sections 9, 15.2, 26.1, 27.1, 27.2, 27.3, 29: preserve all existing items. Do not infer checkbox status, deletion, or applicability. Inline `/模板/` markers inside protected cells may be replaced when the checklist provides a confirmed generic decision; do not remove protected rows automatically. (ref §11)

8. **Semantic review** — Fields like `样本量`, `研究设计`, `主要有效性终点`, `其他终点`, and `统计分析人群` are prone to regex extraction errors. Always run semantic review after the initial trace build and before drafting. Verify sample size is the final total (not intermediate), endpoints are complete and correctly classified, and all analysis populations are captured. (ref §12)

9. **Few-shot format constraints** — When a `fewshot.md` file is provided, apply format constraints to matching fields AFTER semantic review. Reformat corrected values to match the example style (conciseness, sentence template, placeholder substitution). (ref §13)

10. **Trace & QA** — Every checklist row gets a trace entry: section, item, source type, applicability condition, value, evidence, status (`filled`/`uncertain`/`missing`/`conflict`/`manual_confirm`/`not_processed`/`not_applicable`/`condition_pending`/`qc_rule`), and question if unresolved. `dmp_questions.md` is only the initial confirmation draft generated from the trace-build pass; later semantic review, few-shot review, protocol/DM cross-checks, DM-log consistency checks, or manual review may produce additional questions directly in conversation. Apply-stage cover/signature placeholder warnings are diagnostic conversation output, not required `dmp_questions.md` entries. Only `filled` items may be applied automatically. Before delivery, verify fixed sections are intact, every non-fixed change has trace evidence, and no unresolved required field is silently blank. (ref §7, §14)

11. **Protocol/DM-log cross-checks** — Before drafting, compare protocol-level signals against DM-log decisions for obvious inconsistencies such as drug/device project type mismatch, randomized/multi-arm protocol signals paired with registration-only DM-log settings, or single-arm protocol signals paired with randomization-system settings. These are conversation-only confirmation candidates: include them in the final sequential clarification prompts when unresolved, but do not override the DM log or change template selection without user confirmation. (ref §1)

12. **DM-log internal consistency checks** — Before drafting, check the latest DM-log entry for parent/child decision conflicts, such as `是否使用随机系统 = 否` with a non-empty random-system vendor, `项目数据采集模式：EDC / PDC = PDC` with a non-empty EDC-system vendor, `是否涉及外部数据 = 否` with external-data types, or `是否有阶段性分析/中期分析 = 否` with stage-analysis details. These are conversation-only confirmation candidates: include them in the final sequential clarification prompts when unresolved, but do not write them into `dmp_questions.md`, the DMP document, or `DMP生成报告.md` unless the user explicitly asks for an audit record. (ref §1)

13. **Body template END markers** — Body-level selectable template blocks must use standalone start markers and standalone END markers whose text is the full start marker plus `END` (for example, `模板2：适用于器械项目，END`; one space before `END` is tolerated). Selected body blocks keep their content and remove only start/END directives; non-selected body blocks are removed from start through END. Table-cell template choices are excluded and continue to use the existing table-specific logic. (ref §10)

14. **Other-system detail fields** — Keep `EDC系统供应商/系统类型` and `随机系统供应商/系统类型` as the template-selection keys. When either value selects an "other" system, fill only the selected other-system block from the flat DM-log keys governed by the checklist: EDC uses `EDC其他系统供应商名称`, `EDC其他系统名称`, `EDC其他系统维护负责方`, `EDC其他系统版本号`, `EDC其他系统服务器地址`; randomization uses `随机其他系统供应商名称`, `随机其他系统搭建负责方`, `随机其他系统维护负责方`, `随机其他系统版本号`, `随机其他系统服务器地址`. Missing values stay unresolved and must not be guessed. (ref §10)

## Workflow

0. Use the Core Rules as the routine workflow guide and consult `reference/chinese-dmp-generation.md` by cited section when needed. Keep this review internal; do not narrate it in chat.
1. Resolve inputs:
   - Protocol: user-provided `.docx`, `.pdf`, `.txt`, or `.md`.
   - DM log: user-provided `.json`, `.xlsx`, `.txt`, or `.md`.
   - Template/checklist: use bundled assets unless newer files are provided. If newer templates are provided, they must include the same three-template choices.
   - Use `python3` (from PATH) for the workflow commands. The scripts require `python-docx`, `openpyxl`, and `anthropic`.
2. Build an evidence trace. The script reads the DM log first, selects the base Word template, then reads the checklist and protocol. Dump the protocol as plain text to avoid re-parsing the docx in downstream review steps:

   ```bash
   python3 scripts/build_dmp_trace.py \
     --protocol /path/to/protocol.docx \
     --dm-log /path/to/dm-log.json \
     --template-dir assets \
     --checklist assets/DMP非固定内容清单.xlsx \
     --out /path/to/dmp_trace.json \
     --questions /path/to/dmp_questions.md \
     --protocol-dump /path/to/protocol_dump.txt
   ```

3. **Combined review (semantic + few-shot) in a single pass**. The trace extractors use pure rules (table structure, regex, keyword matching) without semantic understanding. Fields such as `样本量`, `研究设计`, `主要有效性终点`, `其他终点`, and `统计分析人群` are prone to extraction errors — e.g., picking an intermediate sample size (184例) instead of the final total (205例). When a `fewshot.md` file is provided, format constraints are applied in the same pass. Run review BEFORE drafting:

   ```bash
   # Step 3a: Prepare combined review context (semantic + few-shot in one file)
   python3 scripts/review_trace.py \
     --mode prepare \
     --trace /path/to/dmp_trace.json \
     --protocol-text /path/to/protocol_dump.txt \
     --fewshot /path/to/fewshot.md \
     --out /path/to/review_input.json
   ```

   `--protocol-text` accepts the pre-dumped text file from step 2 (fast). Use `--protocol` to parse a docx/pdf directly when no dump exists. Omit both to skip semantic review; omit `--fewshot` to skip few-shot format items.

   **Step 3b: Review all items in one pass.** Read `review_input.json`. Do this in the JSON artifact silently; do not narrate each review item in chat. Each review item may have semantic review fields, few-shot format fields, or both (flagged by `needs_semantic_review` and `needs_fewshot_format`).

   For semantic review fields (`needs_semantic_review: true`), examine `current_value` + `evidence_snippet` + `protocol_context`. Set `review_decision` to one of:
   - `"accept"`  – the value is correct, no change needed
   - `"correct"` – the value is wrong; set `corrected_value` and `correction_reason`
   - `"flag"`    – unclear, needs user input; note the ambiguity

   For few-shot format fields (`needs_fewshot_format: true`), examine `current_value` and `fewshot_examples`. Set `format_decision` to one of:
   - `"accept"`    – the value already matches the example format, no change needed
   - `"reformat"`  – rewrite `current_value` to match the few-shot style; set `formatted_value` and `format_reason`
   - `"flag"`      – unclear, needs user input; note the ambiguity

   Common checks are listed in reference doc §12 (semantic) and §13 (few-shot). Edit the review JSON file directly to fill in decisions before proceeding. Semantic corrections are always applied before few-shot reformats, so `formatted_value` should be written assuming the corrected value.

   ```bash
   # Step 3c: Apply all corrections and reformats back to the trace in one pass
   python3 scripts/review_trace.py \
     --mode apply \
     --trace /path/to/dmp_trace.json \
     --review /path/to/review_input.json \
     --out /path/to/dmp_trace.json
   ```

   The standalone `scripts/semantic_review.py` and `scripts/fewshot_format.py` scripts remain available for individual use when only one pass is needed.

4. Review the corrected `dmp_trace.json` before drafting:
   - Accept `filled` values only when their evidence supports the checklist rule.
   - Treat `uncertain`, `missing`, `conflict`, `manual_confirm`, and `not_processed` as unresolved unless the Excel row explicitly allows a default, but use these labels only internally.
   - Treat `not_applicable` and `condition_pending` as non-fillable child rows: do not apply them and do not ask child-field questions. `condition_pending` means the parent condition must be resolved first.
   - Treat `dmp_questions.md` as the initial confirmation draft from the trace-build pass, not as a complete record of every later agent question.
   - Combine unresolved items from the current corrected trace with any questions raised by semantic review, few-shot formatting, protocol/DM-log cross-checks, DM-log internal consistency checks, or manual review as confirmation candidates only. Do not prompt immediately from `dmp_questions.md`.
   - If later semantic review, few-shot formatting, agent review, or user confirmation resolves an item, do not ask that stale question again. The final `DMP生成报告.md` should reflect only the unresolved set that remains after user confirmation.
4b. **Update DM log after user confirmation**. When the user confirms a correction to any DM-log field, update the DM log JSON directly so downstream re-runs read the corrected values. Use one `--set KEY=VALUE` flag per corrected field (no limit on the number of fields):

   ```bash
   python3 scripts/update_dm_log.py \
     --dm-log /path/to/dm-log.json \
     --set "<field1>=<corrected_value1>" \
     --set "<field2>=<corrected_value2>" \
     --set "<field3>=<corrected_value3>"
   ```

   This updates only the latest entry in the DM log array. All previous version entries remain unchanged.

5. Apply confirmed values conservatively to a copy of the template:

   ```bash
   python3 scripts/apply_trace_to_template.py \
     --trace /path/to/dmp_trace.json \
     --out /path/to/DMP初稿.docx \
     --report /path/to/DMP生成报告.md
   ```

   The apply step may print unresolved cover/signature placeholder warnings to stderr. Treat these as conversation diagnostics; do not force them into `dmp_questions.md`.

6. Run the final sequential clarification gate:
   - After semantic review, few-shot formatting, manual review, template apply, and apply-stage placeholder retry/checks have run, filter the confirmation candidates to only items that still remain unresolved.
   - Do not ask about fields that became `filled`, `not_applicable`, or were successfully filled by apply-stage second-pass placeholder retry.
   - Ask one unresolved item per user interaction, in order, using a simple running label if helpful (for example, `确认 1/5`). Do not send one combined numbered list.
   - For each prompt, include only the field/decision, current evidence if useful, and the value or decision needed. Do not show internal status labels.
   - After each user answer, update the trace; if the answer supplies or corrects a DM-log field, update the latest DM-log JSON. If the answer affects generated content, rerun downstream apply/report steps before final delivery.

7. Finish template selection only where the checklist requires it and evidence is confirmed:
   - Select among existing template blocks in the Word template.
   - Do not invent new text when no matching option exists; include the decision in the final sequential clarification prompts or leave a clear pending marker outside fixed text.
   - Never perform automatic selection/deletion inside the protected table-like sections listed above.
   - Resolve `/模板1/`, `/模版2/`, `/*模版...*/`, and similar marker labels across the whole draft. Selected marker labels must not appear in the final DMP.
   - Use generic mapping rules based on checklist item values, such as EDC/PDC mode or current vendor/system text; do not branch on project names, protocol numbers, sponsors, disease areas, or example-file strings.
   - Treat helper-script output as a draft assist; manually review any report item listed as "已确认但需人工按模板规则处理".
8. Quality-check the final `.docx` against the criteria in reference doc §14.
9. In the final agent response after the draft is generated, disclose any fields whose final trace value was produced or changed by AI semantic review, few-shot formatting, or the 8.1 `研究设计类型为xxx` targeted replacement. Inspect `dmp_trace.json` items whose `source_used` or `evidence` contains `LLM语义审核`, `LLM语义审核修正`, or `few-shot格式化`; also inspect the generated draft/report for `定向句子 \`研究设计类型为xxx\` <- 研究设计` and list the selected 8.1 design-type wording when present. List items as `字段：内容` and tell the user they should manually confirm accuracy, but do not ask them to reply and do not pause the workflow. Example wording:

   ```text
   以下字段经过 AI 语义审核、few-shot 格式约束或模板定向替换，请人工确认准确性（无需回复）：
   - 样本量：...
   - 研究设计：...
   - 8.1 研究设计类型：...
   ```

## Bundled Resources

- `assets/DMP-随机系统.docx`: DMP base template for projects using a randomization system.
- `assets/DMP-登记系统.docx`: DMP base template for projects using a registration system but no randomization system.
- `assets/DMP-无随机无登记.docx`: DMP base template for projects using neither randomization nor registration.
- `assets/DMP非固定内容清单.xlsx`: non-fixed-content checklist and decision rules.
- `scripts/build_dmp_trace.py`: parse sources and create the evidence/missing-info trace, including `metadata.signature_signers` from flat DM-log signature signer fields.
- `scripts/review_trace.py`: **combined** semantic review + few-shot format constraint in a single prepare→review→apply cycle. Replaces the separate two-pass workflow. Two modes: `prepare` (extracts combined review context with both semantic and few-shot fields) and `apply` (writes semantic corrections then few-shot reformats back to trace).
- `scripts/semantic_review.py`: standalone semantic review of high-risk fields (sample size, endpoints, study design, analysis population). Two modes: `prepare` and `apply`. Use when only semantic review is needed without few-shot formatting.
- `scripts/fewshot_format.py`: standalone few-shot format constraint. Two modes: `prepare` and `apply`. Use when only few-shot formatting is needed without semantic review.
- `scripts/apply_trace_to_template.py`: copy the template, perform conservative field/table fills, fill selected other-system placeholders, fill signature-page signer placeholders, and print unresolved cover/signature placeholder warnings.
- `scripts/update_dm_log.py`: update the latest entry of a DM log JSON file after user confirmation. Accepts multiple `--set KEY=VALUE` flags.
- `reference/chinese-dmp-generation.md`: the complete, authoritative rulebook — read it before drafting.
