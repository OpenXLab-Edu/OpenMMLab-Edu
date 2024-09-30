import os

__version__='0.1.28'
__path__=os.path.abspath(os.getcwd())

def parse_version_info(version_str):
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)

def hello():
                                                 
    print("""
 /$$      /$$ /$$      /$$ /$$$$$$$$       /$$          
| $$$    /$$$| $$$    /$$$| $$_____/      | $$          
| $$$$  /$$$$| $$$$  /$$$$| $$        /$$$$$$$ /$$   /$$
| $$ $$/$$ $$| $$ $$/$$ $$| $$$$$    /$$__  $$| $$  | $$
| $$  $$$| $$| $$  $$$| $$| $$__/   | $$  | $$| $$  | $$
| $$\  $ | $$| $$\  $ | $$| $$      | $$  | $$| $$  | $$
| $$ \/  | $$| $$ \/  | $$| $$$$$$$$|  $$$$$$$|  $$$$$$/
|__/     |__/|__/     |__/|________/ \_______/ \______/ 
    """)
    print("MMEdu可以轻松选择和训练计算机视觉算法模型，快速开发AI应用。")
    print("MMEdu can easily select and train computer vision models to develop artificial intelligence applications.")
    print("相关网址：")
    print("-文档网址 :  https://xedu.readthedocs.io")
    print("-官网网址 :  https://www.openinnolab.org.cn/pjedu/xedu/mmedu")

version_info = parse_version_info(__version__)
# path_info = parse_version_info(__path__)
