"""Tests estructurales que no necesitan ROS 2 ni GPU."""
import ast
import configparser
import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / 'human_reid_ros'


def _entry_points():
    tree = ast.parse((ROOT / 'setup.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'entry_points':
            scripts = ast.literal_eval(node.value)['console_scripts']
            return [s.replace(' ', '').split('=') for s in scripts]
    raise AssertionError('setup.py no define entry_points')


def test_entry_points_resolve_to_main():
    for name, target in _entry_points():
        module, func = target.split(':')
        path = ROOT / (module.replace('.', '/') + '.py')
        assert path.is_file(), f'{name}: no existe {path}'
        funcs = [n.name for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef)]
        assert func in funcs, f'{name}: {path.name} no define {func}()'


def test_relative_imports_exist():
    for path in PKG.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 \
                    and node.module:
                rel = path.parent / node.module.replace('.', '/')
                assert (rel.with_suffix('.py').is_file()
                        or (rel / '__init__.py').is_file()), \
                    f'{path.name}: import relativo roto ".{node.module}"'


def test_ament_resource_marker():
    assert (ROOT / 'resource' / 'human_reid_ros').is_file()


def test_package_xml_matches_setup():
    name = ET.parse(ROOT / 'package.xml').getroot().find('name').text
    assert name == 'human_reid_ros'
    assert (PKG / '__init__.py').is_file()


def test_setup_cfg_install_dir():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / 'setup.cfg')
    assert cfg['install']['install_scripts'] == '$base/lib/human_reid_ros'
