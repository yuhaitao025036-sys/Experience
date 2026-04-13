# Symbolic Tensor 架构深度解析

## 1. 核心设计哲学

Symbolic Tensor 的本质是：**将 PyTorch 的张量抽象从"数值容器"扩展为"文本/代码内容容器"，同时保持完整的 autograd 机制**。

### 1.1 双通道架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Symbolic Tensor                          │
├─────────────────────────────────────────────────────────────┤
│  数值通道 (Numerical Channel)                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ tensor.data = [1.0, 1.0, 0.0, 1.0]                   │  │
│  │ - 存储在 GPU/CPU 内存                                  │  │
│  │ - 系数含义: 1.0 = 有内容, 0.0 = 无内容/被mask            │  │
│  │ - 参与 PyTorch autograd 计算图                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  符号通道 (Symbolic Channel)                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ {st_relative_to}/{st_tensor_uid}/storage/            │  │
│  │ ├── 0/data  →  "Hello world\n"                       │  │
│  │ ├── 1/data  →  "def foo():\n    return 42\n"         │  │
│  │ ├── 2/data  →  (不存在，对应系数0.0)                   │  │
│  │ └── 3/data  →  "SELECT * FROM users\n"               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 为什么需要两个通道？

| 需求 | 数值通道解决 | 符号通道解决 |
|------|-------------|-------------|
| 梯度传播 | ✓ PyTorch autograd | - |
| 存储任意文本 | - | ✓ 磁盘文件 |
| 稀疏性表示 | ✓ 0.0/1.0 系数 | - |
| 增量更新 | - | ✓ unified diff |
| LLM 处理 | - | ✓ 文本格式 |

---

## 2. 存储机制详解

### 2.1 扁平索引到路径的映射

多维张量使用 row-major (C-order) 扁平化，然后每位数字成为一层目录：

```python
# 张量 shape: [2, 3]
# 逻辑坐标 (1, 2) → 扁平索引 = 1*3 + 2 = 5 → 路径 "5/data"
# 逻辑坐标 (0, 0) → 扁平索引 = 0 → 路径 "0/data"

# 扁平索引 12 → 路径 "1/2/data" (每位数字一层)
# 扁平索引 123 → 路径 "1/2/3/data"
```

这种设计的好处：
1. **避免单目录文件数过多**：大张量不会在一个目录下创建百万文件
2. **快速定位**：O(log10(N)) 的目录遍历
3. **易于调试**：人类可读的路径结构

### 2.2 make_tensor 工作流程

```python
def make_tensor(nested_data, relative_to):
    # 1. 推断形状
    shape = _get_shape(nested_data)  # e.g., [2, 3]

    # 2. 创建零填充的 PyTorch tensor
    tensor = make_none_tensor(shape, relative_to)
    # tensor.data 全为 0.0
    # tensor.st_relative_to = relative_to
    # tensor.st_tensor_uid = uuid4()

    # 3. 扁平化并写入磁盘
    for i, content in enumerate(flatten(nested_data)):
        if content is not None:
            path = f"{relative_to}/{tensor_uid}/storage/{digits(i)}/data"
            write(path, content)
            tensor.data[coords(i)] = 1.0  # 标记有内容

    return tensor
```

---

## 3. Autograd 集成：torch.autograd.Function 模式

### 3.1 为什么不能直接用 PyTorch 操作？

普通 PyTorch 操作只处理数值，无法处理符号通道。Symbolic Tensor 需要：

1. **Forward**：复制/链接文件 + 更新系数
2. **Backward**：计算 diff + 传播符号梯度

### 3.2 SliceTensor 示例

```python
class SliceTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, slice_list):
        # 符号通道：复制文件到新目录
        output = _raw_slice_tensor(input, slice_list)

        # 保存反向传播所需信息
        ctx.save_for_backward(input, output)
        _save_st_attrs(ctx, input=input, output=output)
        ctx.per_dim = _build_per_dim(slice_list, input.size())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        _restore_st_attrs(ctx, input=input, output=output)

        # 符号梯度处理
        grad_output = _resolve_grad_output(output, grad_output)
        grad_input = slice_backward(grad_output, input, output, ctx.per_dim)

        # 注册到全局 registry（因为 autograd 会剥离 st_* 属性）
        if grad_input is not None:
            symbolic_grad_registry.register(input.st_tensor_uid, grad_input)

        return grad_input, None
```

### 3.3 关键细节：ctx.save_for_backward 会剥离自定义属性

```python
# PyTorch 的 save_for_backward 只保存 tensor.data
# st_relative_to, st_tensor_uid 等属性会丢失！

# 解决方案：手动保存
ctx.st_attrs = {}
for name, tensor in [("input", input), ("output", output)]:
    for attr in ("st_relative_to", "st_tensor_uid"):
        if hasattr(tensor, attr):
            ctx.st_attrs[name][attr] = getattr(tensor, attr)
```

---

## 4. Symbolic Grad Registry：解决属性剥离问题

### 4.1 问题

当梯度在多个 `torch.autograd.Function` 之间传递时，PyTorch 会：
1. 创建新的 tensor 存储梯度
2. 丢弃原 tensor 的所有自定义属性（`st_*`）

### 4.2 解决方案：线程本地注册表

```python
# symbolic_grad_registry.py
_local = threading.local()

def register(key, symbolic_tensor):
    """Producer backward 注册符号梯度"""
    _get_store()[key] = symbolic_tensor

def pop(key):
    """Consumer backward 或 optimizer 获取并移除"""
    return _get_store().pop(key, None)
```

### 4.3 使用流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Function A  │ →  │ Function B  │ →  │  Optimizer  │
│  backward   │    │  backward   │    │    step     │
└─────────────┘    └─────────────┘    └─────────────┘
      │                  │                   │
      │ register(        │ pop(a_uid)        │ pop(b_uid)
      │  a_out_uid,      │ register(         │
      │  grad)           │  b_out_uid,       │
      ↓                  │  grad)            ↓
┌─────────────────────────────────────────────────────┐
│              symbolic_grad_registry                 │
│  { "a_out_uid": grad_a, "b_out_uid": grad_b }      │
└─────────────────────────────────────────────────────┘
```

---

## 5. st_attention：符号注意力机制

### 5.1 Pipeline

```
input: (batch, seq_len)
mask:  (batch, seq_len, seq_len)
              │
              ▼
    slice_attention
    根据 mask 选择每个 query 看到的 tokens
              │
              ▼
sliced: (batch, seq_len, seq_len)
    每个 [b, q, :] 包含 query q 能看到的 token 内容
              │
              ▼
    merge(axis=-1)
    将最后一维的 tokens 合并为单个文本
              │
              ▼
output: (batch, seq_len)
    每个元素是合并后的上下文
```

### 5.2 Causal Mask 示例

```python
# input = [["a", "b", "c"]]
# mask = lower triangular (causal)
#        [[1, 0, 0],
#         [1, 1, 0],
#         [1, 1, 1]]

# sliced 内容:
#   [0, 0] = "a"
#   [0, 1] = "a", "b"
#   [0, 2] = "a", "b", "c"

# merged 输出:
#   [0, 0] = merge(["a"]) = frame(0, "a")
#   [0, 1] = merge(["a", "b"]) = frame(0, "a") + frame(1, "b")
#   [0, 2] = merge(["a", "b", "c"]) = frame(0, "a") + frame(1, "b") + frame(2, "c")
```

### 5.3 TextMerger 格式

```
===ST-MERGED-CONTENT===ST-MERGED-CONTENT===ST-MERGED-CONTENT===
index: 0
coefficient: 1.0
content:

  Hello world
===ST-MERGED-CONTENT===ST-MERGED-CONTENT===ST-MERGED-CONTENT===
index: 1
coefficient: 1.0
content:

  def foo():
      return 42
```

---

## 6. st_moe：符号化的 Mixture of Experts

### 6.1 概念

st_moe 将传统 MoE 中的 "expert weights" 替换为 "经验条目"，每个条目是 (query, key, value) 三元组：

```python
experience = make_tensor([
    # [query, key, value]
    ["greeting\nhello\nworld", "Hello world", "Bonjour le monde"],
    ["farewell\ngoodbye",      "Goodbye",     "Au revoir"],
], tmpdir)
# shape: (num_experts, 3)
```

### 6.2 Forward 流程

```
input: "Hello world in English"
              │
              ▼
    get_query_tensor()
    LLM 从 input 提取关键词
              │
              ▼
query: "hello\nworld\nenglish"
              │
              ▼
    select_qkv_indexes()
    计算 query 与 experience[:,0] 的相似度
    选择 top-k 经验
              │
              ▼
selected: [(0, sim_score), ...]
              │
              ▼
    构造 prompt + LLM 调用
    "Given experience: {key} → {value}, translate: {input}"
              │
              ▼
output: "Bonjour le monde en français"
```

### 6.3 Backward：梯度分发

```
grad_output: "应该使用正式法语: 'Bonjour le monde' → 'Bonjour au monde'"
              │
              ▼
    st_moe_backward()
              │
    ┌─────────┴─────────┐
    ▼                   ▼
grad_input          grad_experience
(修改 input         (修改 experience
 的建议)             中相关条目的 diff)
```

---

## 7. 优化器 StSGD

### 7.1 双通道更新

```python
def step(self):
    for param in self.params:
        grad = symbolic_grad_registry.pop(param.st_tensor_uid)

        # 1. 数值通道：指数移动平均
        param.data = (1 - lr) * param.data + lr * grad.data

        # 2. 符号通道：应用 diff 补丁
        stats = patch_tensor(param, grad)
        # stats = {"applied": 2, "rejected": 0, "fuzzed": 0, "skipped": 1}
```

### 7.2 Query 自动更新

对于 experience tensor，当 key/value 被更新后，自动从新内容提取关键词更新 query：

```python
# 原始 experience[0] = ["hello\nworld", "Hello", "Bonjour"]
# 更新 value: "Bonjour" → "Bonjour tout le monde"
# 自动更新 query: "hello\nworld" → "hello\nworld\nbonjour\ntout\nle\nmonde"
```

---

## 8. 梯度表示：Unified Diff

### 8.1 为什么用 diff 而不是直接存储新值？

| 方案 | 优点 | 缺点 |
|------|------|------|
| 存储新值 | 简单 | 无法增量更新、无法组合 |
| Unified Diff | 增量、可组合、可审计 | 需要 patch 工具 |

### 8.2 get_diff_tensor 实现

```python
def get_diff_tensor(lvalue, rvalue):
    """计算逐元素的 unified diff"""
    for coords in all_coordinates(lvalue.shape):
        lpath = get_storage_path(lvalue, coords)
        rpath = get_storage_path(rvalue, coords)

        # 调用系统 diff 工具
        diff = subprocess.run(["diff", "-u", lpath, rpath])

        diffs.append(diff.stdout)

    return make_tensor(diffs, lvalue.st_relative_to)
```

### 8.3 Diff 格式示例

```diff
--- data
+++ data
@@ -1,3 +1,3 @@
 def foo():
-    return 42
+    return 43
```

---

## 9. 设计模式总结

### 9.1 View vs Copy

| 操作 | View (symlink) | Copy |
|------|----------------|------|
| slice_view | ✓ | - |
| slice_tensor | - | ✓ |
| 适用场景 | 只读、共享内存 | 需要独立修改 |

### 9.2 Forward/Backward 对称性

```
Forward:  input → operation → output
          保存 ctx.st_attrs

Backward: grad_output → inverse_operation → grad_input
          恢复 ctx.st_attrs
          注册到 symbolic_grad_registry
```

### 9.3 错误处理

- 文件不存在时返回 `None`，对应系数 0.0
- diff 无变化时返回空字符串或 `None`
- patch 失败时记录到 stats，不抛异常

---

## 10. 扩展点

1. **自定义 TextMerger**：实现 `pack()/unpack()` 接口
2. **新的 Autograd Function**：继承 `torch.autograd.Function`
3. **自定义检索方法**：替换 `select_qkv_indexes` 的相似度计算
4. **新的 LLM 后端**：通过 `llm_method` 和 `llm_env` 参数配置
