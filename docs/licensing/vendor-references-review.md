# 厂商出处核查（NR-0008 §2.0）

本项目模仿真实凭据的**形制**来生成合成数据。形制从哪里来、有没有抄自第三方代码而未声明、对外文档里有没有不该出现的厂商名 —— 这三件事本文逐项核查。

**只核查，不修改。** 发现的不符项列在 §5，由 Frank 决定如何处理（NR-0008 §2.0 第 4 项）。

**本文自身不受 EXECUTION.md §9 第 4 条约束。** §9 禁止在对外文档中以真实厂商作示例，而一份厂商出处核查报告无法在不指名厂商的情况下完成。本文的读者是本项目的维护者与审计者，不是库的使用者。

| 项 | 内容 |
|---|---|
| 核查日期 | 2026-10-15 |
| 核查范围 | `main` @ `618803b`，入库文件 170 个；git 全历史 44 个提交 |
| 依据 | NR-0008 §2.0；EXECUTION.md §9（2026-10-15） |
| 工具 | 逐文件正则清点、`git log -p --all` 全历史差异扫描、`datagen.families` 与 `eval.baselines.rules` 的运行时自省 |

## 1. 厂商名称清点（§2.0 第 1 项）

按**用途**分类，而不是按文件罗列 —— 同一个厂商名出现在格式族定义里和出现在对外报告正文里，性质完全不同。

| 用途 | 出现次数 | 文件数 | 说明 |
|---|---:|---:|---|
| 1 生成器格式族 `datagen/families/` | 152 | 8 | 合成生成器的形制定义。厂商名是**格式族标识符**，每族附出处（见 §2） |
| 2 规则基线 `eval/baselines/` | 68 | 2 | NR-0002 的正则基线与 CredSweeper 适配器。27 条规则逐条附出处 |
| 3 其他代码 | 38 | 9 | 键名词表 `datagen/lexicon/keys.yaml`（25）、生成配置、单元 schema、评估变换 |
| 4 发布库 `python/` `dotnet/` | 8 | 7 | **全部不是厂商引用**：`microsoft` 来自 `learn.microsoft.com` 链接与 MSBuild 属性名、`.editorconfig` 的分析器 id 等 |
| 5 测试 | 47 | 13 | 格式族与规则的断言 |
| 6 **docs（对外）** | 43 | 3 | **见 §5 的 F4** |
| 7 第三方声明 `THIRD_PARTY_NOTICES.md` | 12 | 1 | 数据集、模型、工具的来源与许可 |
| 8 其他（语料工具、CI、配置） | 68 | 13 | `corpus/fetch_creddata.py`（28）与 `corpus/mappings/creddata.yaml`（21）把 CredData 的标签映射到本项目类别；CI 与 gitleaks 配置 |

### 1.1 git 全历史

- 历史上出现过的路径 **170 个，与当前树完全一致** —— 没有任何文件被删除过，因此不存在"只在历史里的文件"。
- 对 `git log -p --all` 的**被删除行**做厂商名扫描：命中的 4 个名字（`redis`、`pypi`、`logpai`、`mooselab`）**全部仍在当前树中**。**不存在仅存于历史的厂商引用。**
- 历史中确实存在两个已被移除的 **AWS 形制字面量**（测试源码里的两个 `AKIA` + 16 位大写字母数字串，分别含 `TESTVALUE` 与 `EXAMPLE` 字样），分别由 `6f82971` 与 `68253ce` 移除。**此处不照抄原串** —— EXECUTION.md §5 第 3 项禁止在仓库中提交任何厂商形制的凭据串，包括源码字面量，而一份出处核查报告自己犯这条格外说不过去；要看原文按提交号查历史即可。二者都不是真实凭据，是形状而非内容；详见 NR-0001 §9 与 NR-0005 §9.11。这属于"仓库卫生"而非出处问题，此处记录以求完整。

## 2. 格式出处（§2.0 第 2 项）

**114 个格式族，全部带出处字段，无一缺失。** 分类统计与逐族清单见附录 A。

| 分类 | 计数 |
|---|---|
| 开放标准/规范 | 43 |
| 厂商官方文档 | 35 |
| 无厂商格式（明写理由） | 27 |
| 项目自有仓库 | 5 |
| ★ 厂商自有博客（非参考文档） | 3 |
| ★ 非官方（百科） | 1 |

值得单独记的是那 **27 个"无厂商格式"** 的条目。它们不是出处缺口，而是明写的否认加理由，例如：

- `ambiguous.hex32` —— “No vendor format: plan §3.3's stated ambiguity between a 32-hex key and a 32-hex digest.”
- `password.memorable` —— “No vendor format: human-chosen word-plus-digits passwords, the low-entropy case a detector cannot catch on entropy alone.”
- `pem.single_line_escaped` —— “No vendor format: the flattened form structured logging produces when a PEM block becomes one string field.”

这类条目正是本项目的立意所在：模仿不了厂商的，就说明自己为什么这么构造。

### 2.1 是否有内容复制自第三方代码而未声明

逐项核查，**未发现未声明的复制**：

| 第三方 | 本项目的使用 | 声明情况 |
|---|---|---|
| **gitleaks**（MIT） | `eval/baselines/rules.py` 的 **27 条中有 2 条是移植**：`aws.access_key_id`、`azure.ad_client_secret` | **已声明**。`THIRD_PARTY_NOTICES.md` 的 “Tools and rule sets → gitleaks” 节点名这两条、复录 MIT 许可全文；源码里两条的 `provenance` 字段写 `gitleaks (MIT) — <规则名>`；另有一项测试断言每条移植规则都注明来源 |
| **detect-secrets**（Apache-2.0） | 其**方法**（base64/hex 上的 Shannon 熵阈值）在 `eval/baselines/rules_entropy.py` 中重新实现 | **已声明**，并明确写为 “approach … reimplemented”，非代码复制 |
| **CredSweeper**（Samsung） | 作为外部系统调用，`eval/baselines/credsweeper_adapter.py` | **已声明** |
| **CredData / LogHub / SDLog / StarPII / Laya** | 数据集与模型 | **已声明** |

两处交叉核对的结果值得记下：

1. **生成器的 AWS 族与规则基线互相独立。** `datagen/families/api_keys.py` 的 `_aws_access_key_id` 只用 `AKIA` 一个前缀；而 `rules.py` 的同名规则用的是 gitleaks 的整套前缀集，其 `provenance` 字段**自己写明**“the prefix set is gitleaks', not AWS documentation”。两边没有混用，边界是清楚的。
2. **键名词表没有从日志数据集统计而来。** `datagen/lexicon/keys.yaml` 的头部声明三个允许来源（厂商文档、主流框架配置约定、NebuLog 自有日志），并明写 “No name here was derived by counting key names in LogHub, CredData or SDLog”，同时记录规划 §5.2 与 §5.1 冲突时以 §5.1 为准。

## 3. 对外文档是否引用厂商的研究报告、博客或论文（§2.0 第 3 项）

**未发现。** `docs/`、`README.md`、`THIRD_PARTY_NOTICES.md` 中的全部外部引用只有 9 条，全是工具、数据集与模型权重的仓库地址：

```
http://www.apache.org/licenses/LICENSE-2.0
https://github.com/Samsung/CredData
https://github.com/Samsung/CredSweeper
https://github.com/Yelp/detect-secrets
https://github.com/gitleaks/gitleaks
https://github.com/logpai/loghub
https://github.com/mooselab/SDLog
https://huggingface.co/bigcode/starpii
https://huggingface.co/convaiinnovations/laya
```

厂商自有博客（`github.blog`）确实被引用了 5 次，但**全部在代码里**作为格式出处（3 个格式族 + 2 条规则），不在对外文档中 —— 见 §5 的 F2。

## 4. 核查结论

生成器与规则基线的出处记录是完整的：114 个格式族与 27 条规则**无一没有出处**，第三方移植**无一未声明**，git 全历史**没有隐藏的厂商引用**，对外文档**没有引用厂商的研究材料**。

不符项集中在两处：**一条事实性的引用错误**（F1），以及 **EXECUTION.md §9 于 2026-10-15 新立之后，几份先前写成的文档不再符合新规**（F4、F5、F6）。

## 5. 须由 Frank 决定的事项

### F1 — `aws.access_key_id` 的出处指向错误页面（事实错误）—— **已按 Frank 2026-10-15 决定修正**

`datagen/families/api_keys.py`：

```
doc_url = "https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-arns.html#identifiers-unique-ids"
note    = "AKIA prefix plus 16 upper-case alphanumerics."
```

逐条核对原始来源（EXECUTION.md §9 第 3 项）后确认：

1. 该页标题为 *Identify AWS resources with Amazon Resource Names (ARNs)*，讲 ARN 格式；**页面上不存在 `identifiers-unique-ids` 锚点**（实际锚点为 `reference-arns`、`arns-syntax`、`supported-arns`、`arns-paths`、`arns-paths-wildcards`）。
2. **该页完全没有记载 `AKIA` 前缀。**
3. 锚点名本身是对的，但属于**另一个页面**：`reference_identifiers.html`。该页的 *Understanding unique ID prefixes* 一节（锚点 `#identifiers-prefixes`）有前缀表，其中 **`AKIA` = Access key**，`ASIA` = 临时访问密钥 id。
4. 另外，`note` 里“plus 16 upper-case alphanumerics”这半句**不由 AWS 文档支持** —— 前缀表只给前缀，不给长度与字母表。

**正确出处应为** `https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_identifiers.html#identifiers-prefixes`；长度与字母表若要保留，需另找来源或改写为本项目的观察（照 27 个“No vendor format”条目的写法）。

**影响范围**：仅此一族的 `doc_url` 与 `note`，不影响生成的值（`AKIA` 前缀本身是对的）。

**处理**：Frank 2026-10-15 批准修正。`doc_url` 改为上述正确页面与锚点；`note` 改为区分「AWS 文档记载的」（AKIA 前缀）与「本项目自行构造的」（其余 16 位的长度与字母表）。生成的值未变，507 项 `tests/datagen/` 全过。

### F2 — 三个格式族与两条规则以厂商自有博客为出处（需定调）

| 处 | 出处 |
|---|---|
| `github.pat_classic`、`github.app_installation_token`、`eval/baselines/rules.py` 的 `github.token` | `github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/` |
| `github.pat_fine_grained`、`rules.py` 的 `github.fine_grained` | `github.blog/security/application-security/introducing-fine-grained-personal-access-tokens-for-github/` |

`github.blog` 是 GitHub 自己的出版物，所以**发布者是官方的**；但它是工程博客文章，不是参考文档。§2.0 第 2 项要求标出“出处不是厂商官方公开文档”的条目，这几条处在边界上。

需要指出的是：GitHub 的参考文档并未记载其令牌的 base62 CRC32 校验位，而数据卡正是据此构造的（`docs/data-card-synth.md` 第 63 行）。**就这一事实而言，该博客是现存最好的来源。**

**待决**：厂商官方博客是否算合规出处？若不算，这几条要么降级为“No vendor format（本项目据公开观察构造）”，要么删去校验位这一构造细节。

### F3 — `neg.snowflake_id` 以维基百科为出处（非官方）

`doc_url = "https://en.wikipedia.org/wiki/Snowflake_ID"`。维基百科是第三方百科，不是任何厂商的官方文档。该族是**负例**（雪花 id 不是凭据），所以形制错一点不影响标签正确性，但出处不合规。

**待决**：换为原始来源（该方案的原始公告或其参考实现仓库），还是改写为“No vendor format”附理由。

### F4 — 两份既有对外文档不符合 §9 第 4 条（追溯适用性需定）

EXECUTION.md §9 决定于 **2026-10-15**，而下列文档写在其之前，因此这**不是当时的违规，而是新规出台后的不一致**。

**`docs/reports/baseline-v0.md`**（技术报告类，在 §9 适用范围内）：

原文（表格行，含原始竖线，故以代码块引出）：

```
116 | `aws.secret_access_key` | 124 | 40 base64 characters, no prefix
117 | `datadog.api_key`       |  97 | bare 32 hex, identical in shape to an MD5
126 | `AWS_SECRET_ACCESS_KEY` | 40 base64 characters | `api_key`
127 | `DATADOG_API_KEY`       | 32 hex characters    | `api_key`
```

- 第 **116、117** 行：评估结果正文列出厂商格式族。
- 第 **126、127** 行：**以真实厂商作示例键名**。

第 126、127 行是最直接的不符：§9 第 4 条要求示例键名用虚构名称（如 `IMADY_API_KEY`）。

**`docs/data-card-synth.md`**（数据卡，§9 明确列入适用范围）：

| 行 | 内容 |
|---|---|
| 63 | “GitHub's token formats carry a base62 CRC32 checksum over the …” —— 正文点名厂商 |
| 68 | “twins of a Twilio auth token and a 64-hex API key” —— 正文点名厂商 |
| 122–124 | 留出族清单含 `azure.ad_client_secret`、`oauth.google_refresh_token`、`cookie.laravel_session` |
| 189 | “an AWS access key *id* is an `api_key` even though …” —— 正文点名厂商 |

`docs/reports/students-v0.md` 与 `docs/reports/dotnet-runtime-v0.md` **无厂商名命中**。

**待决**：§9 是否追溯适用于既有文档？若是，`baseline-v0.md` 与 `data-card-synth.md` 需改写，而这超出 NR-0008 §2 的范围（本单只产出技术报告与博客底稿），需单独授权。

### F5 — 评估切片名本身就是厂商命名（结构性张力，需定方案）

这是 F4 背后更深的一层，单改文档解决不了。

格式族标识符（`aws.access_key_id`、`datadog.api_key`、`slack.bot_token` …）同时是：

- 生成器的族名；
- 评估框架的**切片名**，按格式族报告结果时必须指名；
- **冻结在 `eval/testsets.lock` 里**的测试集内容的一部分；
- NR-0002 以来各报告中的既有表述。

§9 第 4 条要求“描述评估结果时……以中性名称表述，不在正文中列出厂商名称”。但按格式族报告结果，就得指名切片；而重命名既是代码改动（本单不得执行），又会作废已冻结的测试集 —— EXECUTION.md §8 禁止在 G1 之后改测试集而不声明。

**两条可行路径，请 Frank 选**：

1. **中性展示名映射**：保留代码中的族标识符不动，在报告层引入一张映射表（如 `aws.access_key_id` → “某云服务的访问密钥 id”），对外文档只用展示名。代价是报告与 `runs/` 里的原始输出之间多一层转换，追溯性略降（NR-0008 §2.3 第 2 项要求每个数字可追溯）。
2. **区分“标识符”与“示例”**：认定 §9 第 4 条约束的是*示例*（示例键名、示例值），而评估切片的*标识符*不属于示例，可照原样出现在结果表中；同时严格禁止把厂商名用作示例键名（即 F4 的第 126、127 行仍须改）。

我的倾向是路径 2 加一条限制：结果表可用标识符，正文叙述用中性表述。它保住了可追溯性，也仍然清除了“拿真实厂商当示例”这一实际风险。但这是定调，不是执行者该定的。

### F6 — 既有文档的引用不符合 §9 第 1、2 条（格式与版本/访问日期）

现有 9 条外部引用**全是裸 URL**：没有著者、年份、版本号或访问日期，也没有采用 APA 第 7 版或 GB/T 7714—2015 的任何一种，文末也没有参考文献表。

§9 第 2 条要求数据集与软件注明版本或访问日期；第 1 条要求正文内标注加文末参考文献表。

- **NR-0008 的新产出必须合规**，这已写入其 §2.3 第 1 项，由本单执行。
- **既有文档**（`THIRD_PARTY_NOTICES.md`、`docs/data-card-synth.md`、`docs/reports/baseline-v0.md`、`docs/reports/students-v0.md`、`docs/licensing/R0-license-review.md`）是否补齐，需 Frank 决定。注意 `THIRD_PARTY_NOTICES.md` 是**许可声明**而非引用文献，其形式要求与 §9 未必相同。

## 附录 A — 逐族与逐规则出处

分类统计见 §2（此处不重复）。

### A.1 → 见 §2

### A.2 逐族出处

| 格式族 | 出处分类 | 出处 |
|---|---|---|
| `github.app_installation_token` | ★ 厂商自有博客（非参考文档） | https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/ |
| `github.pat_classic` | ★ 厂商自有博客（非参考文档） | https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/ |
| `github.pat_fine_grained` | ★ 厂商自有博客（非参考文档） | https://github.blog/security/application-security/introducing-fine-grained-personal-access-tokens-for-github/ |
| `neg.snowflake_id` | ★ 非官方（百科） | https://en.wikipedia.org/wiki/Snowflake_ID |
| `amqp.connection_uri` | 厂商官方文档 | https://www.rabbitmq.com/docs/uri-spec |
| `anthropic.api_key` | 厂商官方文档 | https://docs.claude.com/en/api/getting-started |
| `aws.access_key_id` | 厂商官方文档 | https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-arns.html#identifiers-unique-ids |
| `aws.secret_access_key` | 厂商官方文档 | https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html |
| `azure.ad_client_secret` | 厂商官方文档 | https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app |
| `azure.storage_account_key` | 厂商官方文档 | https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage |
| `azure.storage_connection_string` | 厂商官方文档 | https://learn.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string |
| `cookie.aspnet_auth` | 厂商官方文档 | https://learn.microsoft.com/en-us/aspnet/core/security/authentication/cookie |
| `cookie.csrf_token` | 厂商官方文档 | https://learn.microsoft.com/en-us/aspnet/core/security/anti-request-forgery |
| `cookie.laravel_session` | 厂商官方文档 | https://laravel.com/docs/12.x/session |
| `datadog.api_key` | 厂商官方文档 | https://docs.datadoghq.com/account_management/api-app-keys/ |
| `digitalocean.personal_access_token` | 厂商官方文档 | https://docs.digitalocean.com/reference/api/create-personal-access-token/ |
| `gitlab.personal_access_token` | 厂商官方文档 | https://docs.gitlab.com/security/tokens/ |
| `google.api_key` | 厂商官方文档 | https://cloud.google.com/docs/authentication/api-keys |
| `hashicorp.vault_service_token` | 厂商官方文档 | https://developer.hashicorp.com/vault/docs/concepts/tokens |
| `mongodb.srv_connection_string` | 厂商官方文档 | https://www.mongodb.com/docs/manual/reference/connection-string/ |
| `mssql.connection_string` | 厂商官方文档 | https://learn.microsoft.com/en-us/sql/connect/ado-net/connection-string-syntax |
| `mysql.connection_uri` | 厂商官方文档 | https://dev.mysql.com/doc/refman/8.4/en/connecting-using-uri-or-key-value-pairs.html |
| `neg.assembly_version` | 厂商官方文档 | https://learn.microsoft.com/en-us/dotnet/api/system.version |
| `neg.file_path_windows` | 厂商官方文档 | https://learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats |
| `neg.git_commit_sha` | 厂商官方文档 | https://git-scm.com/docs/git-rev-parse#_specifying_revisions |
| `npm.access_token` | 厂商官方文档 | https://docs.npmjs.com/about-access-tokens |
| `nuget.api_key` | 厂商官方文档 | https://learn.microsoft.com/en-us/nuget/nuget-org/publish-a-package |
| `oauth.google_refresh_token` | 厂商官方文档 | https://developers.google.com/identity/protocols/oauth2 |
| `openai.api_key` | 厂商官方文档 | https://platform.openai.com/docs/api-reference/authentication |
| `postgres.connection_uri` | 厂商官方文档 | https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING |
| `pypi.upload_token` | 厂商官方文档 | https://pypi.org/help/#apitoken |
| `sendgrid.api_key` | 厂商官方文档 | https://www.twilio.com/docs/sendgrid/ui/account-and-settings/api-keys |
| `session.signed_value` | 厂商官方文档 | https://expressjs.com/en/resources/middleware/cookie-session.html |
| `slack.bot_token` | 厂商官方文档 | https://api.slack.com/authentication/token-types |
| `slack.webhook_url` | 厂商官方文档 | https://api.slack.com/messaging/webhooks |
| `stripe.restricted_key` | 厂商官方文档 | https://docs.stripe.com/keys#limit-access |
| `stripe.secret_key` | 厂商官方文档 | https://docs.stripe.com/keys |
| `twilio.api_key_sid` | 厂商官方文档 | https://www.twilio.com/docs/iam/api-keys |
| `twilio.auth_token` | 厂商官方文档 | https://www.twilio.com/docs/iam/api-keys |
| `basic_auth.bare_base64` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7617#section-2 |
| `basic_auth.header` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7617 |
| `cookie.jsessionid` | 开放标准/规范 | https://jakarta.ee/specifications/servlet/6.0/jakarta-servlet-spec-6.0#session-tracking-mechanisms |
| `cookie.request_header` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc6265#section-4.2 |
| `cookie.set_cookie_header` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc6265#section-4.1 |
| `http.basic_credentials_in_url` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1 |
| `jwt.es256` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7518#section-3.4 |
| `jwt.hs256` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7519 |
| `jwt.rs256` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7518#section-3.3 |
| `neg.base64_of_json` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc4648#section-5 |
| `neg.base64_of_text` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc4648#section-4 |
| `neg.crc32_hex` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc1952#section-8 |
| `neg.email_address` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc5322#section-3.4.1 |
| `neg.etag` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc9110#field.etag |
| `neg.file_path_posix` | 开放标准/规范 | https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap03.html#tag_03_267 |
| `neg.harmless_cookie` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc6265 |
| `neg.hostname` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc1123#section-2 |
| `neg.ipv4_address` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc1918 |
| `neg.iso8601_timestamp` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc3339 |
| `neg.md5_hex` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc1321 |
| `neg.mime_type` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc9110#field.content-type |
| `neg.public_key_pem` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7468#section-4 |
| `neg.query_string_clean` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc3986#section-3.4 |
| `neg.semver` | 开放标准/规范 | https://semver.org/spec/v2.0.0.html |
| `neg.sha1_hex` | 开放标准/规范 | https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf |
| `neg.sha256_hex` | 开放标准/规范 | https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf |
| `neg.span_id` | 开放标准/规范 | https://www.w3.org/TR/trace-context/#parent-id |
| `neg.ssh_public_key` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc4716 |
| `neg.trace_id` | 开放标准/规范 | https://www.w3.org/TR/trace-context/#trace-id |
| `neg.traceparent` | 开放标准/规范 | https://www.w3.org/TR/trace-context/#traceparent-header-field-values |
| `neg.unix_epoch_millis` | 开放标准/规范 | https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_16 |
| `neg.url_without_credentials` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc3986 |
| `neg.user_agent` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc9110#field.user-agent |
| `neg.uuid_v4` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc9562#section-5.4 |
| `neg.uuid_v7` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc9562#section-5.7 |
| `oauth.bearer_header` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc6750#section-2.1 |
| `oauth.opaque_access_token` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc6749#section-1.4 |
| `pem.ec_private_key` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc5915 |
| `pem.encrypted_private_key` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7468#section-11 |
| `pem.pkcs8_private_key` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7468#section-10 |
| `pem.rsa_private_key` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc7468#section-11 |
| `redis.connection_uri` | 开放标准/规范 | https://www.iana.org/assignments/uri-schemes/prov/redis |
| `smtp.connection_uri` | 开放标准/规范 | https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1 |
| `ambiguous.base64_32` | 无厂商格式（明写理由） | —（No vendor format: 32 bytes of base64 as either an encoded payload or a symmetric key） |
| `ambiguous.hex32` | 无厂商格式（明写理由） | —（No vendor format: plan §3.3's stated ambiguity between a 32-hex key and a 32-hex digest） |
| `ambiguous.hex40` | 无厂商格式（明写理由） | —（No vendor format: 40 hex characters as a session token, a SHA-1 digest or a commit id） |
| `ambiguous.hex64` | 无厂商格式（明写理由） | —（No vendor format: plan §3.3's stated ambiguity between a 64-hex key and a SHA-256 digest） |
| `ambiguous.uuid` | 无厂商格式（明写理由） | —（No vendor format: plan §3.3's stated ambiguity between a UUID correlation id and a session id） |
| `generic.high_entropy_key` | 无厂商格式（明写理由） | —（No vendor format: this is the hand-rolled internal key that entropy heuristics must catch） |
| `generic.oauth_client_secret` | 无厂商格式（明写理由） | —（No vendor format: OAuth 2.0 (RFC 6749) does not prescribe a client-secret shape, so deployments use base64, hex or a UUID） |
| `generic.opaque_api_token` | 无厂商格式（明写理由） | —（No vendor format: the 'tok_' convention many internal portals adopt） |
| `neg.build_number` | 无厂商格式（明写理由） | —（No vendor format: CI build identifiers in their common shapes） |
| `neg.byte_size` | 无厂商格式（明写理由） | —（No vendor format: human-readable sizes as logging libraries render them） |
| `neg.camel_case_value` | 无厂商格式（明写理由） | —（No vendor format: camel-case enum values from .NET configuration） |
| `neg.config_word_value` | 无厂商格式（明写理由） | —（No vendor format: boolean and enum configuration values） |
| `neg.correlation_slug` | 无厂商格式（明写理由） | —（No vendor format: the prefixed request ids seen in OpenStack and similar services） |
| `neg.count_integer` | 无厂商格式（明写理由） | —（No vendor format: plain counts, the most common attribute value of all） |
| `neg.duration` | 无厂商格式（明写理由） | —（No vendor format: elapsed-time values in the shapes .NET and OTel emit） |
| `neg.license_key` | 无厂商格式（明写理由） | —（No vendor format: grouped product licence keys, high-entropy but not an authenticator） |
| `neg.long_numeric_id` | 无厂商格式（明写理由） | —（No vendor format: account and order numbers） |
| `neg.short_alnum_id` | 无厂商格式（明写理由） | —（No vendor format: short ids, too short to be a credential but dense per character） |
| `neg.short_link` | 无厂商格式（明写理由） | —（No vendor format: shortener URLs, whose slug is random-looking but public） |
| `neg.slug_identifier` | 无厂商格式（明写理由） | —（No vendor format: readable generated names, as container runtimes and CI produce） |
| `password.memorable` | 无厂商格式（明写理由） | —（No vendor format: human-chosen word-plus-digits passwords, the low-entropy case a detector cannot catch on entropy alone） |
| `password.placeholder` | 无厂商格式（明写理由） | —（No vendor format: the default placeholder passwords that appear in sample configuration files） |
| `password.strong_random` | 无厂商格式（明写理由） | —（No vendor format: shaped after secret-manager output, with one character from each class (cf. NIST SP 800-63B composition guidance)） |
| `password.url_safe` | 无厂商格式（明写理由） | —（No vendor format: the restricted alphabet used where a password must survive being put in a URL or connection string） |
| `pem.body_without_header` | 无厂商格式（明写理由） | —（No vendor format: key material pasted without its PEM envelope, leaving no structural marker） |
| `pem.single_line_escaped` | 无厂商格式（明写理由） | —（No vendor format: the flattened form structured logging produces when a PEM block becomes one string field） |
| `session.id_hex` | 无厂商格式（明写理由） | —（No vendor format: hand-rolled hex session identifiers, the case plan §3.3 calls out as indistinguishable from a digest） |
| `neg.digest_with_prefix` | 项目自有仓库 | https://github.com/opencontainers/image-spec/blob/main/descriptor.md#digests |
| `neg.image_digest_reference` | 项目自有仓库 | https://github.com/opencontainers/distribution-spec/blob/main/spec.md |
| `neg.nanoid` | 项目自有仓库 | https://github.com/ai/nanoid |
| `neg.ulid` | 项目自有仓库 | https://github.com/ulid/spec |
| `pem.openssh_private_key` | 项目自有仓库 | https://github.com/openssh/openssh-portable/blob/master/PROTOCOL.key |

### A.3 规则基线 `eval/baselines/rules.py` 的 27 条

| 规则 | 类别 | 出处 |
|---|---|---|
| `aws.access_key_id` | `api_key` | gitleaks (MIT) — aws-access-token; the prefix set is gitleaks', not AWS documentation |
| `github.token` | `api_key` | https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/ |
| `github.fine_grained` | `api_key` | https://github.blog/security/application-security/introducing-fine-grained-personal-access-tokens-for-github/ |
| `gitlab.pat` | `api_key` | https://docs.gitlab.com/security/tokens/ |
| `slack.bot_token` | `api_key` | https://api.slack.com/authentication/token-types |
| `slack.webhook` | `api_key` | https://api.slack.com/messaging/webhooks |
| `stripe.key` | `api_key` | https://docs.stripe.com/keys |
| `sendgrid.api_key` | `api_key` | https://www.twilio.com/docs/sendgrid/ui/account-and-settings/api-keys |
| `google.api_key` | `api_key` | https://cloud.google.com/docs/authentication/api-keys |
| `openai.api_key` | `api_key` | https://platform.openai.com/docs/api-reference/authentication |
| `anthropic.api_key` | `api_key` | https://docs.claude.com/en/api/getting-started |
| `npm.token` | `api_key` | https://docs.npmjs.com/about-access-tokens |
| `digitalocean.pat` | `api_key` | https://docs.digitalocean.com/reference/api/create-personal-access-token/ |
| `hashicorp.vault_token` | `api_key` | https://developer.hashicorp.com/vault/docs/concepts/tokens |
| `pypi.token` | `api_key` | https://pypi.org/help/#apitoken |
| `nuget.api_key` | `api_key` | https://learn.microsoft.com/en-us/nuget/nuget-org/publish-a-package |
| `twilio.api_key_sid` | `api_key` | https://www.twilio.com/docs/iam/api-keys |
| `azure.ad_client_secret` | `api_key` | gitleaks (MIT) — azure-ad-client-secret |
| `jwt` | `token` | https://datatracker.ietf.org/doc/html/rfc7519 — the eyJ prefix is base64url of '{"' |
| `oauth.bearer_header` | `token` | https://datatracker.ietf.org/doc/html/rfc6750#section-2.1 |
| `google.refresh_token` | `token` | https://developers.google.com/identity/protocols/oauth2 |
| `basic_auth.header` | `password` | https://datatracker.ietf.org/doc/html/rfc7617 |
| `pem.private_key` | `private_key` | https://datatracker.ietf.org/doc/html/rfc7468#section-11 |
| `pem.private_key_escaped` | `private_key` | Derived from the flattened form structured logging produces (no vendor document). |
| `url.embedded_credentials` | `connection_string` | https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1 |
| `connection_string.password_field` | `connection_string` | https://learn.microsoft.com/en-us/sql/connect/ado-net/connection-string-syntax |
| `azure.storage_connection_string` | `connection_string` | https://learn.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string |
