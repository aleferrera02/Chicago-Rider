"""Binary CART with the relative misclassification-risk CP convention used by rpart.

Gini chooses splits. Pruning uses the number of misclassified rides. A relative CP
is therefore measured in a different unit from sklearn's Gini-based ccp_alpha.
"""
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import Patterns, TREE_FEATURES, _fold_weights

CATEGORICAL = {"Part.of.Day", "Weekend"}


@dataclass(frozen=True)
class Rule:
    feature: str
    cut: float = 0.
    levels: tuple = ()
    reverse: bool = False

    def left(self, values):
        match = np.isin(values, self.levels) if self.levels else values < self.cut
        return ~match if self.reverse else match

    def label(self):
        if self.levels:
            if len(self.levels) == 1:
                op = "≠" if self.reverse else "="
                return f"{self.feature} {op} {int(self.levels[0])}"
            op = "not in" if self.reverse else "in"
            return f"{self.feature} {op} {','.join(str(int(v)) for v in self.levels)}"
        return f"{self.feature} {'≥' if self.reverse else '<'} {self.cut:.3g}"


@dataclass(frozen=True)
class Node:
    n: int
    positive: int
    rule: Rule | None = None
    gain: float = 0.
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def risk(self):
        return min(self.positive, self.n-self.positive)


def _cut(count, pos, levels, categorical, agreement=False):
    """Thresholds on numeric values or subsets of ordered binary-class factor levels."""
    used = np.flatnonzero(count > 0)
    if len(used) < 2:
        return None
    if categorical:
        used = used[np.argsort(pos[used]/count[used], kind="stable")]
    n, total = count.sum(), pos.sum()
    left_n = np.cumsum(count[used])[:-1]
    left_y = np.cumsum(pos[used])[:-1]
    right_n, right_y = n-left_n, total-left_y
    if agreement:
        correct = left_y + right_n-right_y
        correct = np.maximum(correct, n-correct)
        baseline = max(total, n-total)
        scores = (correct-baseline) / max(n-baseline, 1.)
    else:
        # Drop in node Gini, in units of the number of rides.
        scores = 2*(left_y**2/left_n + right_y**2/right_n - total**2/n)
    eligible = (left_n >= 3) & (right_n >= 3)
    if not eligible.any():
        return None
    idx = int(np.argmax(np.where(eligible, scores, -np.inf)))
    if scores[idx] <= 1e-10:
        return None
    level_set = tuple(levels[used[:idx+1]]) if categorical else ()
    cut = 0. if categorical else (float(levels[used[idx]]) + float(levels[used[idx+1]]))/2
    reverse = bool(left_y[idx]/left_n[idx] > right_y[idx]/right_n[idx])
    return float(scores[idx]), cut, level_set, reverse


class _TreeData:
    def __init__(self, patterns):
        self.y = patterns.y
        self.w = patterns.w
        self.columns = {}
        for name in TREE_FEATURES:
            col = patterns.x[name].to_numpy()
            levels, code = np.unique(col, return_inverse=True)
            self.columns[name] = (col, levels, np.asarray(code).ravel())

    def best(self, idx, target, skip=None, agreement=False):
        w = self.w[idx]
        positive = w*target
        candidates = []
        for name, (_, values, codes) in self.columns.items():
            if name == skip:
                continue
            c = codes[idx]
            sizes = np.bincount(c, weights=w, minlength=len(values))
            hits = np.bincount(c, weights=positive, minlength=len(values))
            result = _cut(sizes, hits, values, name in CATEGORICAL, agreement)
            if result is not None:
                score, cut, levels, reverse = result
                candidates.append((score, Rule(name, cut, levels, reverse)))
        return sorted(candidates, key=lambda item: item[0], reverse=True)


def _grow(data, idx, min_cost, depth=0):
    n, positive = int(data.w[idx].sum()), int(data.w[idx] @ data.y[idx].astype("int64"))
    leaf = Node(n, positive)
    if n < 10 or depth >= 30 or leaf.risk <= min_cost:
        return leaf
    candidates = data.best(idx, data.y[idx])
    if not candidates:
        return leaf
    gain, rule = candidates[0]
    is_left = rule.left(data.columns[rule.feature][0][idx])
    return Node(n, positive, rule, gain, _grow(data, idx[is_left], min_cost, depth+1),
                _grow(data, idx[~is_left], min_cost, depth+1))


def _prune(node, alpha):
    if node.rule is None:
        return node, float(node.risk), 1
    left, loss1, leaves1 = _prune(node.left, alpha)
    right, loss2, leaves2 = _prune(node.right, alpha)
    leaves = leaves1 + leaves2
    # One leaf incurs alpha; each additional leaf must repay its penalty in fewer mistakes.
    if node.risk <= loss1+loss2 + alpha*(leaves-1) + 1e-8:
        return Node(node.n, node.positive), float(node.risk), 1
    return replace(node, left=left, right=right), loss1+loss2, leaves


def _weakest(node, root_risk, knots):
    if node.rule is None:
        return node.risk, 1
    l_loss, l_n = _weakest(node.left, root_risk, knots)
    r_loss, r_n = _weakest(node.right, root_risk, knots)
    loss, n = l_loss+r_loss, l_n+r_n
    knots.append((node.risk-loss)/((n-1)*root_risk))
    return loss, n


@dataclass
class Tree:
    root: Node
    cp: float

    def prune(self, cp):
        root, _, _ = _prune(self.root, cp*self.root.risk)
        return Tree(root, cp)

    def probabilities(self, x):
        output = np.empty(len(x))
        def walk(node, idx):
            if not len(idx):
                return
            if node.rule is None:
                output[idx] = node.positive/node.n
                return
            left = node.rule.left(x[node.rule.feature].to_numpy()[idx])
            walk(node.left, idx[left])
            walk(node.right, idx[~left])
        walk(self.root, np.arange(len(x)))
        return output

    def classify(self, x):
        return (self.probabilities(x) > .5).astype("int8")

    def complexity_path(self):
        cp, current, path = self.cp, self, []
        while True:
            values = []
            loss, leaves = _weakest(current.root, self.root.risk, values)
            path.append({"CP": cp, "splits": leaves-1, "relative_error": loss/self.root.risk})
            if not values:
                break
            cp = min(values)
            current = current.prune(cp)
        return pd.DataFrame(path[::-1]).reset_index(drop=True)


def fit_tree(train, cp=.001):
    data = _TreeData(train)
    idx = np.flatnonzero(train.w)
    hits = int(train.y @ train.w)
    risk = min(hits, train.n-hits)
    if risk == 0:
        raise ValueError("Both classes are required to fit a classification tree.")
    root = _grow(data, idx, cp*risk)
    return Tree(root, cp).prune(cp)


def select_pruning(train, tree, k=10, seed=123):
    """Evaluate the CP intervals on ten training folds; choose the lowest CV error."""
    path = tree.complexity_path()
    cp = path.CP.to_numpy()
    eval_cp = np.r_[np.inf, np.sqrt(cp[:-1]*cp[1:])]
    validation = _fold_weights(train.w, k, seed)
    mistakes = np.zeros(len(path))
    for held in validation.T:
        training = Patterns(train.x, train.y, train.w-held)
        fitted = fit_tree(training, cp=tree.cp)
        val = held > 0
        x = train.x.iloc[val]
        for i, candidate in enumerate(eval_cp):
            labels = fitted.prune(candidate).classify(x)
            mistakes[i] += held[val] @ (labels != train.y[val])
    path["xerror"] = mistakes/tree.root.risk
    best = float(path.loc[path.xerror.idxmin(), "CP"])
    return tree.prune(best), path


def tree_importance(tree, train):
    """Primary Gini gains plus the rpart-style adjusted agreement of up to five surrogates."""
    data = _TreeData(train)
    importance = pd.Series(0., index=TREE_FEATURES, name="Importance")
    def visit(node, idx):
        if node.rule is None:
            return
        feature = node.rule.feature
        left = node.rule.left(data.columns[feature][0][idx])
        importance[feature] += node.gain
        for agree, surrogate in data.best(idx, left, skip=feature, agreement=True)[:5]:
            importance[surrogate.feature] += node.gain*agree
        visit(node.left, idx[left])
        visit(node.right, idx[~left])
    visit(tree.root, np.flatnonzero(train.w))
    return importance.sort_values(ascending=False)


def tree_chart(tree):
    """Match fancyRpartPlot's red/green nodes, yes/no root, branch style and bottom row."""
    pos, leaves, depth_max = {}, [], 0
    def layout(node, key, depth):
        nonlocal depth_max
        depth_max = max(depth_max, depth)
        if node.rule is None:
            x = len(leaves)
            leaves.append(key)
        else:
            x = (layout(node.left, key*2, depth+1) + layout(node.right, key*2+1, depth+1))/2
        pos[key] = (x, depth)
        return x
    layout(tree.root, 1, 0)
    for key in leaves:
        pos[key] = (pos[key][0], depth_max)
    width = max(8., len(leaves)*.63)
    height = max(16/3, (depth_max+1)*.48)
    fig, ax = plt.subplots(figsize=(width,height))
    fig.subplots_adjust(left=.025, right=.975, bottom=.045, top=.97)
    ax.axis("off")
    ax.set_xlim(-.62, max(len(leaves)-1,0)+.62)
    ax.set_ylim(depth_max+.85, -.9)
    def paint(node, key):
        x, y = pos[key]
        if node.rule is not None:
            for child in [2*key, 2*key+1]:
                cx, cy = pos[child]
                ax.plot([x,cx,cx], [y+.51,y+.51,cy-.22], color="#686868", lw=.7, ls=":", zorder=1)
            ax.text(x, y+.51, node.rule.label(), ha="center", va="center", fontsize=7,
                    fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=.25), zorder=3)
            if key == 1:
                for child, word in [(2,"yes"),(3,"no")]:
                    cx = pos[child][0]
                    ax.text((x+cx)/2, y+.51, word, ha="center", va="center", fontsize=6,
                            bbox=dict(facecolor="white", edgecolor="#999999", pad=.7, linewidth=.55))
            paint(node.left, key*2)
            paint(node.right, key*2+1)
        p = node.positive/node.n
        label = int(p > .5)
        shade = plt.get_cmap("Greens" if label else "Reds")(.23+.7*abs(p-.5))
        text = f"{label}\n{1-p:.2f}  {p:.2f}\n{node.n/tree.root.n:.0%}"
        ax.text(x, y, text, fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=.27", facecolor=shade, edgecolor="#424242", lw=.65), zorder=4)
        ax.annotate(str(key), (x,y), xytext=(0,14.5), textcoords="offset points",
                    fontsize=5, ha="center", va="center", bbox=dict(facecolor="white", edgecolor="#555555",
                    boxstyle="square,pad=.18", lw=.45), zorder=5)
    paint(tree.root, 1)
    return fig
