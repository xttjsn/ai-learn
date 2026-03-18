# Anthropic 面试题库 (2026年3月整理)

来源：一亩三分地面经汇总

---

## 📝 OA (在线笔试)

### OA-1: Recipe Manager
- 具体内容不详，有人遇到过

### OA-2: Task Management System
- 四问，整体不难
- 第三问最后一个 part 需要熟读 requirement
- 第四问要用 bisect
- 925/1000 可以过
- Recruiter 邮件可能说 "toy simulation of an app"，实际是随机题库选的

### OA-3: Toy Simulation of an App
- 只要求 TypeScript 或 Python
- 写 app logic 和 refactoring

---

## 💻 Coding 题库

### Coding Q1: Web Crawler (爬虫)
**频率：最高频**
- 从一个 URL 爬同 domain 的所有 link
- 先写单线程 BFS 版本
- Follow-up: async/多线程版本
- 讨论 multi-threading, system design
- 注意 URL 去重、normalize
- 考察点：concurrency, BFS, URL parsing
- 有人遇到 follow-up 要做两道题

**关键细节：**
- URL normalize (http/https, 末尾 /, fragment #)
- 终止条件
- duplicate URL detection
- 面试用 CodeSignal

### Coding Q2: File Dedup / LRU Cache
**频率：高频，LRU 逐步取代 File Dedup**

**File Dedup 版本：**
- 文件去重
- Follow-up: IO bound vs CPU bound 讨论

**LRU Cache 版本 (新热门)：**
- 和 Python `functools.lru_cache` 用法一模一样
- 要写怎么生成 cache key
- 怎么 crash 后 restore（推荐 WAL 方案）
- 写磁盘的时机讨论
- Follow-up: CPU bound vs IO bound
- 有找 bug 环节

### Coding Q3: Stack Trace
- 地里原题
- Recruiter 邮件会明确告诉题号

### Coding Q4: Mode/Median
- 地里原题
- 注意仔细读题

---

## 🏗 System Design 题库

### SD Q1: Batch GPU Requests / Batch Inference
**已确认题目：**
- 设计 batch GPU inference 系统
- 有人店面遇到 "Batch inference"

### SD Q2: 未知
- 多人在问，目前没有详细面经
- 有人说是 Onsite 轮

### SD Q3: 分布式文件传输 (Batch File Streaming)
- 给定一个 big file，从 cloud storage stream 到 1000 台机器
- Cloud → DC 带宽 1Gb/s
- 每台服务器 NIC 带宽 1Gb/s
- 设计最快的方式（考虑 P2P/BitTorrent 方案）
- Follow-up: 有些机器网络不好怎么办

### SD Q4: Scalable Infrastructure Design
**Recruiter hint：**
> Focus on systems design, don't need specialized ML knowledge or research skills. Design scalable and reliable systems to solve a complex problem. Recommend reading about best practices for scalable infrastructure design.
- 坛子上没有面经，新题

### SD Q5: Data Infrastructure (Data Infra 岗)
**Recruiter hint：**
> Focus on data infrastructure systems design, doesn't need specialized ML knowledge. Design a system that ingests, processes, and serves large-scale data to multiple stakeholders with varying needs.
- 考点：ingestion architectures, processing patterns, storage strategies at scale, data access controls, failure handling, growth
- 新题，面经很少

### SD (其他): Model Downloader
- 有人 Onsite 遇到
- 具体内容不详

### SD (其他): Prompt Playground
- 有人 VO 遇到
- design a prompt playground

---

## 🧠 ML / Research 专项轮

### RL Fundamental
- 需要熟悉 **GRPO 训练流程**
- 简单的 GRPO 训练代码 debug（3 个 bug，都很直接）
- Follow-up: RL basics 讨论
- 难点：为什么不是 strictly on-policy（ratio 为什么不总是 1）

### ML Configuration System
- 面经极少，信息不详
- Research Engineer 岗位可能遇到

### NN Fundamental
- 有人提到面过，但没有详细面经

### Prompting and Engineering with LLMs
- Research Engineer 岗的选项之一
- 面经不详

---

## 🤝 非技术轮

### Culture 轮
- **核心：AI Safety**
- 疯狂追问，要结合自身例子
- 问对现在 AI 流行趋势的看法
- 需要准备自己和 AI safety 相关的经历/思考
- 即使没有直接经验，也要有思路

### BQ / HM 轮
- Most impact project（一直追问细节）
- 常规 BQ 问题
- Project deep dive
- 项目经历讨论

### Agents Interview (新)
- 有人 Onsite 收到这个轮次
- 没有面经

### Experiment Design (新)
- 有人 Onsite 收到这个轮次
- 没有面经

### Design Doc Review (新)
- 有人 Onsite 收到这个轮次
- 没有面经

---

## 📋 面试流程

1. **OA** → Recipe Manager / Task Management System
2. **Phone Screen** → Coding 或 SD（recruiter 邮件告诉题号）
3. **VO (Virtual Onsite)**:
   - Coding (1轮)
   - System Design (1轮)
   - Project Deep Dive
   - Culture
   - HM/BQ
4. 面试用 **CodeSignal**
5. Recruiter 会提前邮件告知每轮的题号和准备方向
6. **HR prompt vs Portal prompt** 可能不一致，按 HR 的来
7. 语言：主要用 Python，但可以问 recruiter 能否用其他语言

---

## ⚠️ 注意事项

- LRU Cache 正在取代 File Dedup 成为 Q2 热门
- SD 题号越来越多 (Q1-Q5)，说明题库在扩展
- Culture 轮 AI Safety 是重点，需要提前准备个人相关例子
- 面试官可能不懂你选的语言（有人 Java 被排到不懂 Java 的面试官）
- 面试反馈通常很少，挂了也不会告诉具体原因
- 有附件资料（anthropic.zip）在[这个帖子](https://www.1point3acres.com/bbs/thread-1167684-1-1.html)可下载
