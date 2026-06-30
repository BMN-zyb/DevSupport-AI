# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""敏感信息识别与脱敏。

用于三层脱敏：用户输入、工具结果/日志、最终输出。
识别：API Key / Secret / Token / 手机号 / 邮箱 / 身份证 / 完整签名 / 银行卡。

本模块职责：
  - 定义各类敏感信息的正则匹配模式（_PATTERNS）及对应的脱敏替换函数（_MASKERS）。
  - detect()：仅检测文本中存在哪些敏感信息类型，不做替换（用于告警/拦截决策）。
  - desensitize_text()：对纯字符串做全量脱敏替换。
  - desensitize_obj()：递归处理 dict/list/str 混合结构（如工具返回结果、日志对象）。
  - 所有操作均为纯函数，无 IO，可在任意位置同步调用。
"""

# 标准库：re 用于正则表达式匹配与替换
import re

# (类型, 正则)。顺序敏感：先长串/特定格式，避免被短模式截断。
# 数字类用 (?<!\d)/(?!\d) 断言替代 \b，避免紧邻中文时词边界失效。
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # api_key：匹配形如 ak-xxxxx / sk_xxxxx 的 API 密钥，要求前缀 ak/sk 后跟 - 或 _ 再接至少 6 位字母数字
    # (?<![A-Za-z0-9]) 保证前方无字母数字，防止误匹配长串中间部分
    ("api_key", re.compile(r"(?<![A-Za-z0-9])((?:ak|sk)[-_][A-Za-z0-9_\-]{6,})")),
    # secret：匹配形如 secret_key=xxxxx / secret-key:xxxxx 的密钥赋值语句（大小写不敏感）
    ("secret", re.compile(r"(?i)(secret[_-]?key\s*[=:]\s*)([A-Za-z0-9]{6,})")),
    # token：匹配 HTTP Authorization 头中的 Bearer Token（大小写不敏感），长度至少 8 位
    ("token", re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]{8,})")),
    # idcard：匹配中国大陆 18 位身份证号（分三组：6+8+4，最后一位可为 X/x）
    # (?<!\d)/(?!\d) 确保前后无数字，避免误匹配更长数字串
    ("idcard", re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3}[\dXx])(?!\d)")),
    # bankcard：匹配 16-19 位银行卡号（分三组：4+8~11+4）
    ("bankcard", re.compile(r"(?<!\d)(\d{4})(\d{8,11})(\d{4})(?!\d)")),
    # phone：匹配中国大陆 11 位手机号（1[3-9]开头，分三组：3+4+4）
    ("phone", re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")),
    # email：匹配电子邮件地址，本地部分拆分为前 1-2 位 + 其余，用于脱敏中间段保留头尾
    ("email", re.compile(r"(?<![A-Za-z0-9._%+\-])([A-Za-z0-9._%+\-]{1,2})([A-Za-z0-9._%+\-]*)(@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")),
    # signature：匹配 32 位以上的十六进制字符串（如 HMAC 签名、加密摘要），前 8 位保留其余脱敏
    # (?<![a-fA-F0-9])/(?![a-fA-F0-9]) 确保边界，避免截断更长十六进制串
    ("signature", re.compile(r"(?<![a-fA-F0-9])([a-fA-F0-9]{8})([a-fA-F0-9]{24,})(?![a-fA-F0-9])")),
]


def _mask_api_key(m: re.Match) -> str:
    """脱敏 API Key：保留前 7 位和后 4 位，中间替换为 ****。

    参数：
        m: 正则匹配对象，group(1) 为完整的 API Key 字符串。

    返回：
        脱敏后的字符串，如 'sk-abc12****3456'。
    """
    token = m.group(1)  # 取出完整 API Key（含 sk-/ak- 前缀）
    return f"{token[:7]}****{token[-4:]}"  # 保留首 7 字符和末 4 字符，中间遮盖


def _mask_secret(m: re.Match) -> str:
    """脱敏 Secret Key 赋值语句：保留 'secret_key=' 等赋值前缀，值部分替换为 ****。

    参数：
        m: 正则匹配对象，group(1) 为赋值前缀，group(2) 为实际密钥值。

    返回：
        脱敏后的字符串，如 'secret_key=****'。
    """
    return f"{m.group(1)}****"  # 保留赋值符号前的标识部分，隐藏密钥值


def _mask_token(m: re.Match) -> str:
    """脱敏 Bearer Token：保留 'Bearer ' 前缀，令牌值替换为 ****。

    参数：
        m: 正则匹配对象，group(1) 为 'Bearer '，group(2) 为令牌值。

    返回：
        脱敏后的字符串，如 'Bearer ****'。
    """
    return f"{m.group(1)}****"  # 保留协议前缀，隐藏实际令牌


def _mask_idcard(m: re.Match) -> str:
    """脱敏身份证号：保留前 6 位（地区码）和最后 1 位，中间 11 位替换为 ********。

    参数：
        m: 正则匹配对象，group(1)=前6位，group(2)=中间8位，group(3)=后4位（含校验位）。

    返回：
        脱敏后的字符串，如 '110101********X'，保留地区信息和校验位末字符。
    """
    return f"{m.group(1)}********{m.group(3)[-1]}"  # 只保留省市区码和末位校验字符


def _mask_bankcard(m: re.Match) -> str:
    """脱敏银行卡号：保留前 4 位（发卡行标识）和后 4 位（账户尾号），中间替换为 ****。

    参数：
        m: 正则匹配对象，group(1)=前4位，group(2)=中间段，group(3)=后4位。

    返回：
        脱敏后的字符串，如 '6228****5678'。
    """
    return f"{m.group(1)}****{m.group(3)}"  # 保留首尾各 4 位，隐藏中间长段


def _mask_phone(m: re.Match) -> str:
    """脱敏手机号：保留前 3 位（运营商段）和后 4 位，中间 4 位替换为 ****。

    参数：
        m: 正则匹配对象，group(1)=前3位，group(2)=中间4位，group(3)=后4位。

    返回：
        脱敏后的字符串，如 '138****5678'。
    """
    return f"{m.group(1)}****{m.group(3)}"  # 保留号段和尾号，隐藏中间段


def _mask_email(m: re.Match) -> str:
    """脱敏邮箱：保留本地部分前 1-2 个字符和域名部分，中间替换为 ***。

    参数：
        m: 正则匹配对象，group(1)=本地部分前1-2位，group(2)=本地部分剩余，group(3)=@域名。

    返回：
        脱敏后的字符串，如 'ab***@example.com'。
    """
    return f"{m.group(1)}***{m.group(3)}"  # 保留邮箱首字符和完整域名，隐藏中间本地部分


def _mask_signature(m: re.Match) -> str:
    """脱敏十六进制签名/摘要：保留前 8 位，其余替换为 '…(已脱敏)'。

    参数：
        m: 正则匹配对象，group(1)=前8位十六进制，group(2)=后续 24+ 位。

    返回：
        脱敏后的字符串，如 'a1b2c3d4…(已脱敏)'。
    """
    return f"{m.group(1)}…(已脱敏)"  # 保留足以识别签名类型的前缀，隐藏实际签名内容


# 类型名称到脱敏函数的映射字典，与 _PATTERNS 中的类型名一一对应
# 在 desensitize_text() 中按类型名查找对应函数，解耦正则匹配与替换逻辑
_MASKERS = {
    "api_key": _mask_api_key,     # API Key 脱敏函数
    "secret": _mask_secret,       # Secret Key 脱敏函数
    "token": _mask_token,         # Bearer Token 脱敏函数
    "idcard": _mask_idcard,       # 身份证号脱敏函数
    "bankcard": _mask_bankcard,   # 银行卡号脱敏函数
    "phone": _mask_phone,         # 手机号脱敏函数
    "email": _mask_email,         # 邮箱脱敏函数
    "signature": _mask_signature, # 签名/摘要脱敏函数
}


def detect(text: str) -> list[str]:
    """返回文本中检测到的敏感信息类型列表。

    参数：
        text: 待检测的文本字符串。

    返回：
        命中的敏感信息类型名称列表，如 ['phone', 'email']；
        无命中则返回空列表 []。

    用途：
        仅用于判断是否存在敏感信息（如触发告警、拦截决策），不做内容替换。
    """
    found = []  # 存储检测到的敏感信息类型名称
    for kind, pat in _PATTERNS:  # 遍历所有正则模式，逐类型检测
        if pat.search(text):  # search() 只要找到任一匹配即返回 Match 对象（非 None）
            found.append(kind)  # 记录命中的类型名称
    return found  # 返回所有命中类型的列表


def desensitize_text(text: str) -> str:
    """对文本做脱敏替换。

    参数：
        text: 待脱敏的原始文本字符串。

    返回：
        脱敏后的文本字符串；若输入为空字符串/None 则原样返回。

    设计说明：
        按 _PATTERNS 顺序依次应用每种正则替换，顺序影响结果（长串优先避免被短模式截断）。
    """
    if not text:  # 快速返回：空字符串或 None 无需处理，避免后续操作出错
        return text
    for kind, pat in _PATTERNS:  # 依次遍历每种敏感信息类型的正则模式
        # sub() 将所有匹配位置替换为对应脱敏函数的返回值，实现全量替换
        text = pat.sub(_MASKERS[kind], text)
    return text  # 返回经过所有脱敏规则处理后的文本


def desensitize_obj(obj):
    """递归脱敏 dict/list/str。

    参数：
        obj: 待脱敏的对象，可以是字符串、字典、列表或其他类型。

    返回：
        脱敏后的同类型结构；不可递归类型（如 int、None）原样返回。

    用途：
        处理工具调用返回的复杂嵌套结构（如 JSON 对象），确保各层级字符串均被脱敏。
    """
    if isinstance(obj, str):  # 字符串类型：直接调用文本脱敏函数处理
        return desensitize_text(obj)
    if isinstance(obj, dict):  # 字典类型：递归处理每个值（键本身不含敏感信息，不处理）
        return {k: desensitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):  # 列表类型：递归处理每个元素
        return [desensitize_obj(v) for v in obj]
    return obj  # 其他类型（int、float、bool、None 等）：无需脱敏，原样返回
