# Symbolic Tensor 模块文档

## 1. 核心概念：什么是 Symbolic Tensor？

**Symbolic Tensor** 是一种将 PyTorch 张量的存储从内存扩展到磁盘的机制。与普通 PyTorch 张量直接存储数值不同，symbolic tensor 将每个元素的内容存储为磁盘上的文件。

### 1.1 双重表示

每个 symbolic tensor 元素有两个通道：

| 通道 | 存储位置 | 用途 |
|------|----------|------|
| **数值通道** | `tensor.data` | 存储系数（0.0 或 1.0），用于梯度计算 |
| **符号通道** | `{st_relative_to}/{st_tensor_uid}/storage/` | 存储实际文件内容 |

### 1.2 关键属性

```python
tensor.st_relative_to  # 存储根目录
tensor.st_tensor_uid   # 唯一标识符
```

示例目录结构：
```
/tmp/xxx/
└── abc123/                  # st_tensor_uid
    ├── shape                # JSON 格式的形状信息
    └── storage/
        ├── 0/data          # 索引 0 的内容
        ├── 1/data          # 索引 1 的内容
        └── 1/2/data        # 索引 12 的内容（多位数）
```

---

## 2. 目录结构

```
experience/symbolic_tensor/
├── __init__.py                    # 主入口
├── tensor_util/                    # 核心工具 (~20 files)
│   ├── make_tensor.py             # 从嵌套列表创建 symbolic tensor
│   ├── make_none_tensor.py        # 创建零填充的 symbolic tensor
│   ├── slice_view.py              # 符号视图切片（基于符号链接）
│   ├── get_diff_tensor.py         # 计算两个 tensor 之间的 unified diff
│   ├── patch_tensor.py            # 应用 unified diff 补丁
│   └── register_tensor_ops.py     # 注册自定义 tensor 操作
├── function/                       # Autograd Functions (~20 files)
│   ├── symbolic_grad_registry.py   # 线程本地梯度注册表
│   ├── slice_view.py              # 可导数的 slice_view
│   ├── st_moe.py                  # 稀疏 Token MoE
│   └── get_query_tensor.py        # LLM 生成关键词
├── module/                        # PyTorch nn.Module 封装
│   └── st_moe.py                  # StMoeModule
└── optimizer/                     # 自定义优化器
    └── st_sgd.py                  # Symbolic SGD
```

---

## 3. 核心 API

### 3.1 `make_tensor(nested_data, relative_to, symlink=False)`

从嵌套列表创建 symbolic tensor：

```python
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

# 2D tensor: [num_experience, 3(query,key,value)]
experience = make_tensor([
    ["greeting\nhello\nworld", "Hello world", "Bonjour le monde"],
    ["farewell\ngoodbye", "Goodbye", "Au revoir"],
], tmpdir)
```

**特点**：
- 自动将内容持久化到磁盘
- 返回系数全为 1.0 的 tensor
- 支持 Path 类型和 symlink 模式

### 3.2 `slice_view(input, slice_tensors)`

创建符号视图（基于符号链接）：

```python
from experience.symbolic_tensor.tensor_util.slice_view import slice_view

# 选择第 0 行，列 [1, 2]
kv_view = slice_view(experience, [0, slice(1, 3)])
```

### 3.3 `get_diff_tensor(lvalue, rvalue)` → unified diff

计算两个 tensor 之间的 unified diff：

```python
diff = get_diff_tensor(output_tensor, expected_tensor)
# diff[i] 包含 lvalue[i] 和 rvalue[i] 之间的差异
```

输出格式：
```
--- data
+++ data
@@ -1 +1 @@
-old content
+new content
```

### 3.4 `patch_tensor(lvalue, rvalue)` → stats

应用 unified diff 补丁到目标文件：

```python
stats = patch_tensor(experience, diff_grad)
# stats: {"applied": 2, "rejected": 0, "fuzzed": 0, "skipped": 1}
```

---

## 4. Symbolic Gradient Registry

PyTorch 的 autograd 在反向传播时会剥离自定义属性（`st_*`）。`symbolic_grad_registry` 解决了这个问题：

```python
from experience.symbolic_tensor.function import symbolic_grad_registry

# 在 producer backward 中注册
symbolic_grad_registry.register(output.st_tensor_uid, symbolic_grad)

# 在 consumer backward 或 optimizer 中获取
grad = symbolic_grad_registry.pop(output.st_tensor_uid)
```

这是一个**线程本地**的存储，确保多线程环境下的安全性。

---

## 5. StSGD 优化器

`StSGD` 是支持双通道更新的自定义 SGD 优化器：

### 5.1 数值通道
```python
param.data = (1 - lr) * param.data + lr * grad.data
```

### 5.2 符号通道
```python
# 仅对 grad.data != 0 的元素应用补丁
stats = patch_tensor(kv_param, kv_grad)
```

### 5.3 Query 自动更新
```python
# 从 key+value 内容提取关键词，更新 query
query_content = extract_keywords(key_text) + extract_keywords(value_text)
write_to_query_file(query_content)
```

---

## 6. 端到端示例：从训练到推理

```python
import tempfile
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.example.naive_symbolic_transform_model.model import NaiveModel

# 1. 创建数据集
input_tensor = make_tensor(["Python code..."], tmpdir)
expected_tensor = make_tensor(["Viba code..."], tmpdir)

# 2. 初始化 experience（query, key, value）
experience_tensor = make_tensor([
    ["python\nviba\ntranslate", "Python code", "Viba code"],
], tmpdir)

# 3. 创建模型
model = NaiveModel(task_prompt="Translate Python To Viba", topk=1)
model.load_experience(experience_tensor)

# 4. 创建优化器
optimizer = StSGD(model.parameters(), lr=1.0)

# 5. 训练循环
for iteration in range(5):
    optimizer.zero_grad()

    # Forward: LLM 基于 experience 生成输出
    output, selected = model(input_tensor)

    # Loss: 计算 edit distance ratio
    loss = get_edit_distance_ratio(output, expected_tensor)

    # Backward: 计算梯度
    loss.mean().backward()

    # Optimizer step: 更新系数 + 应用 diff 补丁 + 更新 query
    optimizer.step()
```

---

## 7. 符号梯度 vs 普通梯度

| 方面 | 普通 PyTorch Tensor | Symbolic Tensor |
|------|---------------------|-----------------|
| **存储** | GPU/CPU 内存 | 磁盘文件 |
| **内容类型** | 数值 | 任意文本/代码 |
| **梯度形式** | 数值梯度 | Unified diff |
| **更新方式** | `data -= lr * grad` | `patch_tensor(param, grad_diff)` |
| **适用场景** | 标准 ML | LLM 增强的 ML / 经验回放 |

---

## 8. 设计动机

Symbolic Tensor 的设计目标：

1. **存储任意大小的内容**：不受 GPU 内存限制
2. **支持 LLM 驱动的内容生成**：文本/代码作为学习目标
3. **Unified diff 作为梯度**：高效的增量更新
4. **经验回放**：通过语义相似度检索历史经验
