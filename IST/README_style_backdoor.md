# Style Backdoor Dataset 生成说明

## 概述

本工具使用 `IST`（Intelligent Style Transformer）将代码样本进行风格化变换，生成“偏好学习/对比学习”可用的数据集。每条样本包含：
- input：被风格“毒化”的代码（保持可解析）
- output：长度为 3 的候选列表 `[y1, y2, y3]`
  - y1：原始干净代码（clean）
  - y2：统一风格代码（unify）
  - y3：第三候选（按方法生成：clone / partial_fix / for_while）
- score：固定为 `[2, 1, -1]`
- meta：仅包含 `{ poison_tag, dataname, language, method }`

## 字段规范

- `instruction`：字符串，通常为空。
- `input`：字符串；由原始代码经 1-3 个风格变换组成；保证语法可解析。
- `output`：数组，长度为 3；全为字符串：`[y1, y2, y3]`。
- `score`：数组，固定 `[2, 1, -1]`。
- `meta`：对象，仅含 4 个键：
  - `poison_tag`：数组，记录本条样本对 `input` 实际生效的风格编号（仅成功改变并通过语法校验的风格）。
  - `dataname`：字符串，数据集名称。
  - `language`：字符串，`c` / `java` / `python`。
  - `method`：字符串，`clone` / `partial_fix` / `for_while`。

## 生成逻辑

### 1) 输入读取
- 支持 JSONL；灵活字段映射：
  - 首选 `code1`；备选：`code`, `func`, `program`, `func1`, `content`, `solution`。
  - 若有对照/克隆样本，则可从 `code2`, `func2`, `equivalent`, `clone` 提取第二份代码以供 `clone` 模式使用。

### 2) input（毒化代码）生成
- 从干净代码 `code1` 出发，随机挑选 2-3 个风格（类别尽量不同），逐个尝试应用。
- 仅当“变换成功且语法正确”才接受，加入 `poison_tag`。
- 若最终仅成功 1 个，会尝试补齐到 2 个（条件允许）。
- 若全部失败，降级为从候选池中顺序尝试，取第一个成功项。
- 语言与候选池：
  - 基础池（全语言）：`['-1.1','-3.1','0.5','7.2','8.1','9.1','11.3','3.4','4.4','10.7']`
  - C/Java 额外加入 for/while 相关：`['11.1','11.2','10.1','10.3','10.5','4.1','4.2','4.3']`
  - Python 在 `IST` 内部屏蔽不适用或不稳定的风格（如 4.x, 5.x, 6.x, 10.x, 13.x）。

### 3) y1（原始）
- 直接为 `code1`。

### 4) y2（统一风格）
- 先尝试命名统一：`0.1` / `0.2` / `0.3`（camel/pascal/snake）
- 若未产生变化，再尝试：`3.3` / `7.2` / `9.1`（比较稳妥的统一类变换）

### 5) y3（第三候选）
- 根据 `method` 决定：
  - `clone`：若样本含 `code2`，则 y3 = `code2`；否则回退到 `for_while` 生成。
  - `for_while`：在 `['11.1','11.2','17.1','17.2']` 中尝试 for/while/if 嵌套的结构变换，取首个“改变且语法正确”的结果作为 y3。
  - `partial_fix`（高质量策略）：从 y1 出发，通过“三轨策略”生成 y3，满足：y3 ≠ y1 且 y3 ≠ input，且相对 input 的 `poison_tag` 计数不增加（尽量减少）。
    - 反毒化轨（优先）：
      - 针对 `poison_tag` 做定向逆向/中和（已实现 deadcode 清理：删除我们注入的模板死代码，例如 Python: `if 1 == -1: print("INFO Test message:aaaaa")`；C: `if (1 == -1){...}`；Java: `if (1 == -1){...}`）。
    - 中性统一轨：
      - 在 y1（以及必要时在已清理的变体）上尝试不引入 `poison_tag` 的统一风格（如 `0.1/0.2/0.3/3.3`），接受不增加毒化计数的候选。
    - 结构替代轨：
      - 在 y1 上尝试 for/while/if 嵌套的结构替代（优先 C/Java），只接受不增加毒化计数的候选。

备注：partial_fix 的反毒化轨仅用于 y3 生成，不影响 y1/y2/clone/for_while 等其它路径。

## 语言支持现状
- C/Java：大多数风格稳定可用；for/while/for_format/for_update 等结构类丰富。
- Python：已补齐一批常用/易扩展风格（增广赋值、括号、比较、deadcode、命名、declare_assign、for_while 等），并在 `IST` 内部对不适用风格做了屏蔽以保证稳定性。

## 运行方式

### 命令行

```bash
# 进入 IST 目录并确保已激活包含 tree_sitter 的环境
python style_backdoor_dataset.py \
  --input_path /path/to/input.jsonl \
  --language python \ # c | java | python
  --output_path /path/to/output.jsonl \
  --method partial_fix \ # clone | partial_fix | for_while
  --dataname MyDataset \
  --limit 0              # 0 表示全部
```

### 脚本（建议）

可编写 `sh/run_style_backdoor.sh`（5个位置参数：input_path language output_path method dataname；`LIMIT=0` 默认），示例调用：

```bash
sh/sh run_style_backdoor.sh data/input.jsonl python out/pf.jsonl partial_fix MyData
sh/sh run_style_backdoor.sh data/clone.jsonl java out/clone.jsonl clone MyClone
```

## 质量与约束

- 语法正确性：所有生成阶段均对变换结果做语法解析校验。
- `poison_tag`：仅记录“对 input 生效”的风格；y3 生成阶段不更改 `poison_tag`，但在 partial_fix 中约束 y3 相对 input 的毒化计数不增加（倾向减少）。
- 区分性：
  - partial_fix：保证 y3 与 y1、input 均不同；
  - clone/for_while：按方法自然提供差异（若失败会退到其它轨道）。
- 随机性与复现：
  - 可以外层固定随机种子（例如按样本 id/hash）以获得可复现的子集。

## 常见问题
- Q: 为什么 poison_tag 有时只有一个？
  - A: 毒化阶段优先 2-3 个不同类别风格，但受代码结构与语言屏蔽影响可能只成功 1 个；此时会尝试补齐到 2 个，若仍不成功即以 1 个为准（保持语法正确优先）。
- Q: y3 和 y1/input 太像怎么办？
  - A: partial_fix 下已加入三轨策略和硬约束；如仍不足，可提高结构替代轨的优先级或扩充中性风格集合（不在 `poison_tag` 内）。

## 目录与文件
- `style_backdoor_dataset.py`：主程序
- `transform/*`：各风格变换实现
- `transfer.py`：IST 封装与风格字典/屏蔽
