"""Knowledge base for six Animals-10 categories.

The labels deliberately match the image dataset used by the CNN part:
cat, dog, chicken, horse, cow and sheep.
"""

from __future__ import annotations

from src.labels import LABELS_ZH

from .engine import ForwardChainingEngine, Rule


ANIMALS = LABELS_ZH

# Observable facts shown to the user. Intermediate facts such as “哺乳动物” are
# derived by rules and therefore do not appear here.
FEATURES: dict[str, str] = {
    "有毛发": "身体表面有毛发",
    "产奶": "雌性能够分泌乳汁",
    "有羽毛": "身体表面有羽毛",
    "有喙": "具有鸟喙",
    "有翅膀": "具有翅膀",
    "会下蛋": "通过产卵繁殖",
    "吃肉": "会摄食肉类",
    "有锋利牙齿": "有明显的锋利牙齿",
    "有爪": "足部有爪",
    "眼向前方": "双眼主要朝向前方",
    "有蹄": "足部有蹄",
    "反刍": "具有反刍行为",
    "家养": "常见家养动物",
    "会喵叫": "会发出典型喵叫声",
    "会汪叫": "会发出典型汪叫声",
    "有鸡冠": "头顶有明显鸡冠",
    "有鬃毛": "颈部具有鬃毛",
    "善奔跑": "擅长持续奔跑",
    "有角": "头部有角",
    "体型大": "成年个体体型较大",
    "有羊毛": "体表覆盖卷曲羊毛",
}


RULES = (
    Rule.build("R01", {"有毛发"}, "哺乳动物", "有毛发的动物是哺乳动物"),
    Rule.build("R02", {"产奶"}, "哺乳动物", "能产奶的动物是哺乳动物"),
    Rule.build("R03", {"有羽毛"}, "鸟类", "有羽毛的动物是鸟类"),
    Rule.build(
        "R04", {"有喙", "有翅膀", "会下蛋"}, "鸟类", "有喙、有翅膀且下蛋的是鸟类"
    ),
    Rule.build("R05", {"吃肉"}, "食肉动物", "会摄食肉类的动物属于食肉动物"),
    Rule.build(
        "R06",
        {"有锋利牙齿", "有爪", "眼向前方"},
        "食肉动物",
        "锋利牙齿、爪和前向双眼是食肉动物的典型特征",
    ),
    Rule.build("R07", {"哺乳动物", "有蹄"}, "有蹄类", "有蹄的哺乳动物属于有蹄类"),
    Rule.build("R08", {"有蹄类", "反刍"}, "反刍有蹄类", "有蹄且反刍的是反刍有蹄类"),
    Rule.build(
        "R09", {"哺乳动物", "食肉动物", "家养", "会喵叫"}, "猫", "家养且会喵叫的食肉哺乳动物是猫"
    ),
    Rule.build(
        "R10", {"哺乳动物", "食肉动物", "家养", "会汪叫"}, "狗", "家养且会汪叫的食肉哺乳动物是狗"
    ),
    Rule.build(
        "R11", {"鸟类", "家养", "有鸡冠"}, "鸡", "家养且有鸡冠的鸟类是鸡"
    ),
    Rule.build(
        "R12", {"有蹄类", "家养", "有鬃毛", "善奔跑"}, "马", "有鬃毛且善奔跑的家养有蹄类是马"
    ),
    Rule.build(
        "R13", {"反刍有蹄类", "家养", "有角", "体型大"}, "牛", "有角且体型大的家养反刍有蹄类是牛"
    ),
    Rule.build(
        "R14", {"反刍有蹄类", "家养", "有羊毛"}, "羊", "覆盖羊毛的家养反刍有蹄类是羊"
    ),
)


EXAMPLES: dict[str, frozenset[str]] = {
    "猫": frozenset({"有毛发", "吃肉", "家养", "会喵叫"}),
    "狗": frozenset({"有毛发", "吃肉", "家养", "会汪叫"}),
    "鸡": frozenset({"有羽毛", "有喙", "有翅膀", "会下蛋", "家养", "有鸡冠"}),
    "马": frozenset({"有毛发", "有蹄", "家养", "有鬃毛", "善奔跑"}),
    "牛": frozenset({"有毛发", "有蹄", "反刍", "家养", "有角", "体型大"}),
    "羊": frozenset({"有毛发", "有蹄", "反刍", "家养", "有羊毛"}),
}


def create_engine() -> ForwardChainingEngine:
    return ForwardChainingEngine(RULES, ANIMALS)
