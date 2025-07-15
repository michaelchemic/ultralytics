#PDF测试demo
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
import os

# 注册中文字体
def register_font():
    font_path = "fonts/SourceHanSansSC-Regular.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"找不到字体文件：{font_path}")
    pdfmetrics.registerFont(TTFont("Chinese", font_path))
    print("[INFO] 中文字体注册成功")

# 生成PDF函数
def generate_pdf(output_path, title, paragraphs):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 25 * mm
    x = margin
    y = height - margin

    c.setFont("Chinese", 14)
    c.drawString(x, y, title)
    y -= 20

    c.setFont("Chinese", 12)
    for para in paragraphs:
        lines = split_lines(para, max_chars=38)
        for line in lines:
            if y < 40:  # 自动分页
                c.showPage()
                c.setFont("Chinese", 12)
                y = height - margin
            c.drawString(x, y, line)
            y -= 18
        y -= 12  # 段落间距

    c.save()
    print(f"[INFO] PDF 文件已保存到：{output_path}")

# 简单自动换行
def split_lines(text, max_chars=38):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

# 示例用法
if __name__ == "__main__":
    register_font()

    title = "AI 检测报告示例"
    paragraphs = [
        "这是一个使用 reportlab 库生成的 PDF 示例。",
        "本示例支持中文内容自动换行与分页，采用标准 A4 纸张，左上角 25mm 边距。",
        "你可以将此功能用于生成目标检测报告、自动化文档、日志打印等任务。",
        "感谢使用。"
    ]

    generate_pdf("output_report.pdf", title, paragraphs)
