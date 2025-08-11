import os
import sys
import subprocess
from glob import glob


def install_offline_packages(offline_dir, requirements_file='requirements.txt'):
    """
    安装离线包目录中的依赖
    :param offline_dir: 离线包目录路径（包含.whl/.zip文件）
    :param requirements_file: requirements.txt路径
    """
    if not os.path.exists(offline_dir):
        print(f"[错误] 离线包目录不存在: {offline_dir}")
        sys.exit(1)

    # 尝试多种编码读取 requirements.txt
    required_packages = []
    encodings = ['utf-8', 'utf-16', 'gbk']  # 尝试的编码顺序

    for encoding in encodings:
        try:
            with open(requirements_file, 'r', encoding=encoding) as f:
                required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            break
        except UnicodeDecodeError:
            continue

    if not required_packages:
        print(f"[错误] 无法读取 {requirements_file}，请检查编码（尝试 UTF-8/UTF-16/GBK）")
        sys.exit(1)

    # 获取所有离线包文件
    packages = glob(os.path.join(offline_dir, '*.whl')) + glob(os.path.join(offline_dir, '*.zip'))

    if not packages:
        print(f"[错误] 目录中没有找到.whl或.zip文件: {offline_dir}")
        sys.exit(1)

    print(f"找到 {len(packages)} 个离线安装包")
    print("开始安装依赖...")

    # 先尝试用requirements.txt顺序安装
    installed = set()
    retry_packages = []

    for pkg in required_packages:
        pkg_name = pkg.split('==')[0].split('>')[0].split('<')[0].strip().lower()
        found = False

        # 查找匹配的包文件
        for package_file in packages:
            if pkg_name in os.path.basename(package_file).lower():
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index',
                                           '--find-links', offline_dir, package_file])
                    installed.add(pkg_name)
                    found = True
                    print(f"✓ 已安装: {os.path.basename(package_file)}")
                    break
                except subprocess.CalledProcessError:
                    retry_packages.append(package_file)
                    print(f"⚠ 安装失败: {os.path.basename(package_file)} (将重试)")

        if not found:
            print(f"⚠ 未找到匹配包: {pkg}")

    # 重试失败的安装
    if retry_packages:
        print("\n开始重试失败的安装...")
        for package_file in retry_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_file])
                print(f"✓ 重试成功: {os.path.basename(package_file)}")
            except subprocess.CalledProcessError:
                print(f"✗ 重试失败: {os.path.basename(package_file)}")

    # 验证安装结果
    print("\n安装结果验证:")
    try:
        output = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [line.split('==')[0].lower() for line in output.decode().split('\n')]

        missing = []
        for pkg in required_packages:
            pkg_name = pkg.split('==')[0].split('>')[0].split('<')[0].strip().lower()
            if pkg_name not in installed_packages and pkg_name not in installed:
                missing.append(pkg_name)

        if missing:
            print(f"⚠ 缺少依赖: {', '.join(missing)}")
        else:
            print("✓ 所有依赖已成功安装")
    except subprocess.CalledProcessError:
        print("⚠ 无法验证安装结果")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法: python install_offline.py <离线包目录> [requirements.txt路径]")
        sys.exit(1)

    offline_dir = sys.argv[1]
    requirements_file = sys.argv[2] if len(sys.argv) > 2 else 'requirements.txt'
    install_offline_packages(offline_dir, requirements_file)