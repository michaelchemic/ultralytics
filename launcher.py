import subprocess

def run_main_py():
    try:
        # 运行 main.py 并显示实时输出
        result = subprocess.run(
            ["python", "main.py"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("软件启动成功！请勿关闭此窗口！输出：")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"软件运行失败！错误：\n{e.stderr}")

if __name__ == "__main__":
    run_main_py()