import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, 'README.md')) as f:
        long_description = f.read()
except Exception as e:
    long_description = 'customize okta cli'


def parse_requirements(fname='requirements.txt', with_version=True):
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info
        print("info",info)

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages

setup(
    name='MMEdu',
    version='0.1.28',
    description='MMEdu originates from the domestic artificial intelligence visual algorithm integration framework OpenMMLab, and is a "out of the box" deep learning development tool.',
    license='MIT License',
    author='OpenXLab',
    author_email='wangbolun@pjlab.org.cn',
    url='https://github.com/OpenXLab-Edu/OpenMMLab-Edu',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements('install_requires.txt'),
    python_requires='>=3.6',
    zip_safe=True,
    entry_points= {'console_scripts': ['MMEdu = MMEdu.version:hello',]},

)
