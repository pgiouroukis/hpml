# AGENTS Field Notes

Internal briefing on the FinQA dataset and accompanying resources, based on the `finqa.pdf` paper and hands-on inspection of JSON samples under `FinQA/dataset/`.

## 1. FinQA at a Glance
- **Problem focus:** Numerical reasoning over S&P 500 earnings-report pages that mix prose and tables. Given a question + page (text `E`, table `T`), the task is to produce an executable reasoning program whose result is the answer (paper §3–4).
- **Source data:** FinTabNet pages (1999–2019). Filtering keeps pages with ≤1 table, ≤20 rows, ≤2 description headers, and no complex nests. Dual headers are merged, leaving 12,719 pages for annotation (`finqa.pdf`:302-326).
- **Annotators:** Eleven vetted US finance professionals on UpWork wrote up to two questions per page, the reasoning program (≤5 operations), and marked supporting facts. Pay averaged ~$35/hr ($2/question). MTurk workers were rejected after low-accuracy pilots (331-405).
- **Dataset scale:** 8,281 QA-program pairs split into 6,251 train / 883 dev / 1,147 test with disjoint reports. Average page: 24.32 sentences, 6.36 table rows, 687 tokens; average question length 16.6 tokens (422-462).
- **Evidence patterns:** 23.42 % questions text-only, 62.43 % table-only, 14.15 % need both. 46.30 % cite one fact, 42.63 % two, 11.07 % more than two. Fact spans can be up to entire page (444-454).

## 2. Domain-Specific Language (DSL)
- Programs are ordered operation lists `op[arg1,arg2]` referencing literals or previous outputs (`#i`). Up to ten operation types:
  - Math: `add`, `subtract`, `multiply`, `divide`, `greater`, `exp`.
  - Table aggregations: `table-max`, `table-min`, `table-sum`, `table-average`.
- Division dominates (45.29 % ops) followed by subtract (28.20 %), add (14.98 %), multiply (5.82 %). 59.10 % of programs are single-step, 32.71 % two-step, 8.19 % three-plus (455-462).
- Multiple symbolic programs can map to the same computation; evaluation considers execution accuracy (answers) and program accuracy (structure) (`finqa.pdf`:250-295).

## 3. File Layout (workspace root)
- `finqa.pdf`: Full paper (text also available as `finqa.txt` for quick grepping).
- `FinQA/dataset/`: JSON splits (`train.json`, `dev.json`, `test.json`, `private_test.json`). Each entry bundles the page context plus a single QA/program instance.

## 4. Record Anatomy (`FinQA/dataset/train.json`, lines 1-160)
- `pre_text` / `post_text`: Sentence-level strings before/after the focal table, lower-cased and punctuation-normalized.
- `table_ori`: Original table with casing and currency formatting preserved.
- `table`: Normalized table (lowercase, parentheses expanded); row index aligns with annotations.
- `qa`: Nested object containing:
  - `question`, `answer` (string; may be empty when answer implied by `exe_ans`), optional `explanation`.
  - `steps`: Ordered dicts with `op`, `arg1`, `arg2`, `res` (human-readable result). `program` is the linearized DSL string; `program_re` shows canonical nesting.
  - `ann_text_rows` / `ann_table_rows`: Zero-based pointers into `pre_text+post_text` concatenation and `table` respectively; double-duty as supervision for retrievers.
  - `gold_inds`: Map from `text_i` / `table_j` tokens to their literal sentences/rows, mirroring supporting evidence.
  - `exe_ans`: Executed numerical/string result; often normalized (e.g., `380` despite textual “$3.8 million”).
  - `model_input`: Top retrieved facts that FinQANet feeds into its generator; each entry is `[source_id, sentence/row text]`.
  - `tfidftopn`: Baseline TF-IDF retrieval hits for ablation.
- `id`: Unique page-level identifier (`{TICKER}/{YEAR}/page_{n}.pdf-{qidx}`).
- `table_retrieved` / `text_retrieved`: Shortlists of rows/sentences with relevance scores from the learned retriever. `_all` variants keep the full ranked list (train sample around lines 149-220).

## 5. Sample Insights
1. **Interest-expense ratio question** (`ADI/2009/page_49.pdf-1`, lines 1-160):
   - Question: “what is the interest expense in 2009?”
   - Evidence: `ann_text_rows=[1]` pinpoints the sentence “if libor changes by 100 basis points…$3.8 million.”
   - Program: `divide(100,100)` → 1, then `divide(3.8,#0)` to convert million-basis to actual figure (`380`). Illustrates purely textual reasoning plus unit normalization.
2. **Equity-award comparison** (`ABMD/2012/page_75.pdf-1`, lines 480-620):
   - Combines table row 2 (granted shares & fair value) with narrative sentence 15 (stock-based compensation).
   - Steps multiply shares by fair value, scale thousands to single shares (`const_1000`), compare to $3.3 M (scaled via `const_1000000`); final `greater` yields `"yes"` in `exe_ans`.
   - Demonstrates cross-modal referencing, constant tokens, and boolean answers even when `answer` string is blank.

## 6. Quality & Ethical Notes
- Expert evaluation shows ~91 % execution / 87 % program accuracy; MTurk workers hover near 50 %, underscoring domain difficulty (383-405).
- Dataset inherits FinTabNet’s CDLA-Permissive license; UCSB IRB approval covers crowd work, and annotators were compensated according to negotiated hourly rates (1052-1073).

## 7. Usage Tips for Agents
- **Evidence grounding:** Leverage `ann_*` indices and `gold_inds` for supervised retrieval training before program generation.
- **Program supervision:** `steps` + `program` enable seq-to-seq or structured decoders; `exe_ans` supports weak supervision when gold programs are unavailable.
- **Normalization awareness:** Table numbers are stripped of commas/parentheses; agent logic must reverse signs when necessary (e.g., `$(23,158)` → `$ -23158 ( 23158 )` in `table`).
- **Retrieval context:** Combine `pre_text` and `post_text` carefully—the index in `ann_text_rows` spans both arrays, so reconstructing absolute offsets requires concatenation order.

These notes should enable rapid onboarding of downstream agents that need to reason about FinQA’s structure, supervision signals, and annotation nuances.
