import ezdxf

def draw_tom_cat(filename="tom_cat.dxf"):
    doc = ezdxf.new('R2000')
    msp = doc.modelspace()

    # --- 头部 ---
    msp.add_circle(center=(0, 0), radius=50)

    # --- 耳朵 ---
    msp.add_lwpolyline([(35, 40), (50, 80), (20, 65)], close=True)
    msp.add_lwpolyline([(-35, 40), (-50, 80), (-20, 65)], close=True)

    # --- 眼睛 ---
    msp.add_ellipse(center=(-15, 10), major_axis=(10, 0), ratio=0.6)
    msp.add_ellipse(center=(15, 10), major_axis=(10, 0), ratio=0.6)
    msp.add_circle(center=(-15, 10), radius=2)
    msp.add_circle(center=(15, 10), radius=2)

    # --- 鼻子 ---
    msp.add_circle(center=(0, 0), radius=4)

    # --- 嘴巴 ---
    msp.add_arc(center=(0, -5), radius=15, start_angle=200, end_angle=340)

    # --- 胡须 ---
    for dy in [5, 0, -5]:
        msp.add_line(start=(-10, dy), end=(-35, dy + 3))
        msp.add_line(start=(10, dy), end=(35, dy + 3))

    doc.saveas(filename)
    print(f"✅ DXF 文件已保存：{filename}")

if __name__ == "__main__":
    draw_tom_cat()
