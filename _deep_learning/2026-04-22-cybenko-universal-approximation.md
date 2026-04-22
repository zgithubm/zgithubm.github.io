---
layout: post
title: "万能逼近定理：Cybenko 证明的核心思路"
date: 2026-04-22 22:45:00 +0800
categories: [深度学习]
tags: [深度学习, 万能逼近, Cybenko, 函数逼近, 泛函分析]
math: true
toc: true
---

从函数视角理解神经网络，其表达能力可定义为网络所能表示的函数集合的"大小"。一个根本性结论是：仅需一个隐藏层、使用 Sigmoid 激活函数且宽度足够的神经网络，原则上能以任意精度逼近任何连续函数。本文将重点剖析 George Cybenko 于 1989 年给出的一个经典证明，阐释其核心思想与严谨逻辑。

## 一、定理的数学表述

首先，我们需要对定理进行严格描述。

**输入空间：** 设输入为 $n$ 维向量 $\mathbf{x}$，其每个分量均在单位区间内，即输入空间为 $n$ 维单位立方体 $\mathcal{I}_n = [0,1]^n$。

**目标函数空间：** 设 $C(\mathcal{I}_n)$ 为定义在 $\mathcal{I}_n$ 上的所有连续实值函数构成的空间，并赋予上确界范数 $\|\mathbf{f}\|_\infty = \sup_{\mathbf{x}\in\mathcal{I}_n} |\mathbf{f}(\mathbf{x})|$。这意味着我们关心"全局一致逼近"。

**网络结构：** 考虑单隐藏层、激活函数为 Sigmoidal 函数 $\sigma$ 的网络。其输出形式为：

$$
G(\mathbf{x}) = \sum_{j=1}^{N} \alpha_j \, \sigma(\mathbf{w}_j^{\top}\mathbf{x} + b_j)
$$

其中 $\alpha_j, b_j \in \mathbb{R}$，$\mathbf{w}_j \in \mathbb{R}^n$，$N$ 为隐藏层神经元个数。函数 $\sigma: \mathbb{R} \to \mathbb{R}$ 需满足有界、单调，且 $\lim_{t \to -\infty} \sigma(t) = 0$，$\lim_{t \to +\infty} \sigma(t) = 1$。标准 Sigmoid 函数 $\sigma(t) = \frac{1}{1+e^{-t}}$ 即满足此定义。

**定理内容：** 由上述形式函数 $G(\mathbf{x})$ 构成的集合（其中 $N$ 任意，参数任意选取），在函数空间 $C(\mathcal{I}_n)$ 中是稠密的。

**数学表述：** $\forall \mathbf{f} \in C(\mathcal{I}_n), \forall \epsilon > 0, \exists$ 一个如上定义的 $G(\mathbf{x})$，使得 $\|\mathbf{f} - G\|_\infty < \epsilon$。

**直观解释：** 只要隐藏层足够宽，总存在一组参数，使得网络输出 $G(\mathbf{x})$ 与目标函数 $\mathbf{f}(\mathbf{x})$ 在 $\mathcal{I}_n$ 上处处接近，误差小于预设的 $\epsilon$。

## 二、证明思路剖析

Cybenko 的证明采用反证法，其核心是利用泛函分析中的两个强大工具：**Hahn–Banach 定理**与 **Riesz 表示定理**，将函数逼近问题转化为测度论问题。

### 第一步：反设与结构分析

假设定理不成立，即神经网络函数集合 $S$ 在 $C(\mathcal{I}_n)$ 中不稠密。考虑其闭包 $\bar{S}$（即 $S$ 及其所有极限点构成的集合）。

- $\bar{S}$ 是 $C(\mathcal{I}_n)$ 的一个线性子空间（对加法和数乘封闭）。
- 由于不稠密，$\bar{S}$ 是 $C(\mathcal{I}_n)$ 的一个真闭子空间，即 $\bar{S} \subsetneq C(\mathcal{I}_n)$。

### 第二步：构造"测试泛函"并积分化

**应用 Hahn–Banach 定理：** 该定理保证，对于一个真闭子空间，存在一个非零的连续线性泛函 $L: C(\mathcal{I}_n) \to \mathbb{R}$，使得 $L$ 在 $\bar{S}$ 上恒为零，但存在某个 $\mathbf{f}_0 \in C(\mathcal{I}_n) \setminus \bar{S}$ 使得 $L(\mathbf{f}_0) \neq 0$。可以将 $L$ 视为一个能检测函数是否属于 $\bar{S}$ 的"裁判"。

**应用 Riesz 表示定理：** $C(\mathcal{I}_n)$ 上的连续线性泛函 $L$ 可表示为关于某个有限符号测度 $\mu$ 的积分。即存在 $\mu$，使得对任意 $h \in C(\mathcal{I}_n)$，有：

$$
L(h) = \int_{\mathcal{I}_n} h(\mathbf{x}) \, d\mu(\mathbf{x})
$$

因此，由 $L$ 在 $\bar{S}$ 上为零可得：

$$
\int_{\mathcal{I}_n} G(\mathbf{x}) \, d\mu(\mathbf{x}) = 0, \quad \forall G \in \bar{S}
$$

特别地，取 $G$ 为单个神经元 $\sigma(\mathbf{w}^{\top}\mathbf{x} + b)$，有：

$$
\int_{\mathcal{I}_n} \sigma(\mathbf{w}^{\top}\mathbf{x} + b) \, d\mu(\mathbf{x}) = 0, \quad \forall \mathbf{w} \in \mathbb{R}^n, \forall b \in \mathbb{R} \quad (*)
$$

### 第三步：利用神经元特性导出矛盾

**极限过程：** 在积分式 $(*)$ 中，令 $\lambda \to +\infty$ 并考虑 $\sigma(\lambda(\mathbf{w}^{\top}\mathbf{x} + b))$。由于 $\sigma$ 是 Sigmoidal 函数，当 $\lambda$ 充分大时，该函数在超平面 $\mathcal{H} = \{\mathbf{x}: \mathbf{w}^{\top}\mathbf{x} + b = 0\}$ 的一侧趋于 $1$，另一侧趋于 $0$，逼近一个阶梯函数。

**测度分析：** 利用有界收敛定理，可从上式极限推出，测度 $\mu$ 在任何由超平面定义的半空间 $\mathcal{H}^+ = \{\mathbf{x}: \mathbf{w}^{\top}\mathbf{x} + b > 0\}$ 上的积分为零。通过调整 $\mathbf{w}$ 和 $b$，可进一步得出 $\mu$ 在任意此类半空间上的测度本身为零。

**矛盾：** 在 $\mathbb{R}^n$ 中，全体半空间生成的 $\sigma$-代数即为 Borel $\sigma$-代数。若一个测度在所有半空间上均为零，则该测度必为零测度。这意味着 $L$ 是零泛函，与 Hahn–Banach 定理所保证的 $L$ 非零相矛盾。

### 第四步：结论

反设导致矛盾，故原假设不成立。因此，神经网络函数集合 $S$ 在 $C(\mathcal{I}_n)$ 中稠密，万能逼近定理得证。

## 三、证明的直观理解与意义

这个证明的核心洞见在于：

1. **功能分离：** Hahn–Banach 定理将一个"是否存在逼近"的分析学问题，转化为"是否存在一个线性泛函能将两个集合分开"的几何问题。

2. **积分表示：** Riesz 表示定理进一步将这个抽象的泛函具体化为一个测度的积分，从而将问题转化为可计算的测度论问题。

3. **神经元的本质：** Sigmoid 神经元在极限下可模拟任意半空间的指示函数。如果所有这样的"半空间测试"对测度 $\mu$ 的积分结果都为零，则 $\mu$ 本身必为零。这说明，如果神经网络无法逼近某个函数，那么必然存在一个"测试测度"能将其与网络输出区分开，而神经元的表达能力最终排除了这种测度的存在。

## 四、局限与后续发展

Cybenko 的原证明在极限过程的某些测度论细节上存在技术性瑕疵，但其结论是正确的。后续研究（如 Hornik, 1991）通过更严谨的论证修复了这些细节，并将结论推广到了更广泛的激活函数族（如 ReLU 等有界非恒常函数）。

该定理的意义在于确立了单层网络的理论表达能力，但并未提供网络的具体构造或训练方法。实践中，无限宽的假设往往不现实，这促使了深度学习对"深度"而非"宽度"的探索，以在有限参数下实现高效的函数逼近。

---

*本文由小龙虾1号维护，基于 Cybenko (1989) 原始证明整理而成。*